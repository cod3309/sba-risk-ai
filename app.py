import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ⚙️ 1. 설정 (가장 먼저 와야 함)
st.set_page_config(page_title="SBA-CBP 실시간 AI 분석기", page_icon="🏦", layout="wide")

COL_SBA_NAICS = 'NAICSCode'
COL_SBA_LOAN  = 'GrossApproval'
COL_CBP_EST   = 'Number of establishments (ESTAB)'

# 🧠 2. 저장된 모델과 핵심 데이터 로드
@st.cache_resource
def load_model():
    return joblib.load('sba_model.pkl')

@st.cache_data
def load_data():
    # return pd.read_csv('sba_app_data.csv')
    return pd.read_csv('sba_app_data_lite.csv')

# 💡 [여기 새로 추가!] 산업코드 목록도 딱 한 번만 만들어서 저장해둡니다!
@st.cache_data
def get_naics_codes(df):
    naics_2 = df['NAICS_2'].dropna().astype(str).unique().tolist()
    naics_6 = df[COL_SBA_NAICS].dropna().astype(str).unique().tolist()
    return sorted(list(set(naics_2 + naics_6)))

try:
    with st.spinner("⚡대시보드를 불러오는 중입니다..."):
        model = load_model()
        df_main = load_data()
except FileNotFoundError:
    st.error("🚨 'sba_model.pkl' 또는 'sba_app_data.csv' 파일이 없습니다. 먼저 `train_model.py`를 실행해주세요!")
    st.stop()

# ---------------------------------------------------------
# 🎨 [추가] 고급스러운 왼쪽 사이드바 구성
# ---------------------------------------------------------
with st.sidebar:
    st.title("🏦 AI 리스크 심사기")
    st.markdown("---")
    st.markdown("미국 중소기업청(SBA)의 대출 데이터와 인구조사국(CBP)의 지역 상권 데이터를 융합하여, **가장 정교한 기업 부도 리스크를 실시간으로 예측**합니다.")
    st.markdown("---")
    st.info(f"**📊 사용된 데이터베이스**\n\n총 **{len(df_main):,}건**의 과거 대출/파산 기록 및 지역 인프라 데이터")
    st.caption("© 2026 SBA AI Dashboard Project")

# ---------------------------------------------------------
# 🖥️ 메인 웹 UI 부분
# ---------------------------------------------------------
st.title("🏦 SBA-CBP 통합 AI 리스크 대시보드")
st.markdown("<br>", unsafe_allow_html=True) # 약간의 여백 추가

tab1, tab2 = st.tabs(["📊 산업별 분석 대시보드", "🔮 실시간 개별 기업 심사"])

# =========================================================
# 탭 1: 대시보드
# =========================================================
with tab1:
    st.subheader("🔎 맞춤형 산업군 리스크 분석")
    
    # 💡 [새로 추가된 로직] 지역(State) 목록 만들기 ('전국' 옵션 포함)
    state_list = ["전국"] + sorted(df_main['State_Full'].dropna().unique().tolist())
    
    # 산업 코드 목록 추출 (기존과 동일)
    all_codes = get_naics_codes(df_main)
    
    # 💡 [디자인 개선] 드롭다운 2개를 나란히 배치합니다.
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        selected_state = st.selectbox(
            "📍 분석할 지역(주)을 선택하세요", 
            options=state_list, 
            index=0, # 기본값은 '전국'
            help="'전국'을 선택하면 미국 전체 데이터를, 특정 주를 선택하면 해당 지역 데이터만 분석합니다."
        )
        
    with col_input2:
        search_code = st.selectbox(
            "🔍 분석할 산업군 코드(NAICS)", 
            options=all_codes,
            index=all_codes.index("54") if "54" in all_codes else 0
        )
        
    search_code = str(search_code).strip() 
    
    # 1. 먼저 산업 코드로 데이터를 필터링합니다.
    if len(search_code) == 2:
        ind_df = df_main[df_main['NAICS_2'].astype(str) == search_code].copy()
        ind_title = f"대분류 산업 [{search_code}]"
    else:
        ind_df = df_main[df_main[COL_SBA_NAICS].astype(str) == search_code].copy()
        ind_title = f"상세 산업 [{search_code}]"
        
    # 2. 💡 [핵심 필터링] 사용자가 특정 지역을 선택했다면, 그 지역 데이터만 한 번 더 추려냅니다!
    if selected_state != "전국":
        ind_df = ind_df[ind_df['State_Full'] == selected_state]
        view_title = f"[{selected_state}] {ind_title}"
    else:
        view_title = f"[전국] {ind_title}"
        
    # 데이터 유무 확인 및 화면 출력
    if ind_df.empty:
        st.warning(f"⚠️ {view_title}에 해당하는 대출 데이터가 없습니다. 다른 지역이나 코드를 선택해 보세요.")
    else:
        st.success(f"✅ {view_title} 데이터를 성공적으로 불러왔습니다. (총 {len(ind_df):,}건)")
        
        display_df = ind_df.sample(n=min(len(ind_df), 1000), random_state=42)
        
        feat_cols = [COL_SBA_LOAN, COL_SBA_NAICS, COL_CBP_EST]
        display_df['Prob'] = model.predict_proba(display_df[feat_cols].fillna(0))[:, 1] * 100
        
        avg_p = display_df['Prob'].mean()
        avg_l = ind_df[COL_SBA_LOAN].mean()

        # 깔끔한 KPI 메트릭
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 평균 대출액", f"${avg_l:,.0f}")
        est_title = "🏢 지역 평균 사업체 수" if selected_state == "전국" else "🏢 해당 지역 총 사업체 수"
        m2.metric(est_title, f"{ind_df[COL_CBP_EST].mean():,.0f}개")
        m3.metric("🚨 AI 예측 평균 실패 확률", f"{avg_p:.1f}%")

        st.divider()

        # 차트 출력
        col_l, col_r = st.columns(2)
        with col_l:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=avg_p,
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]},
                       'steps': [{'range': [0, 30], 'color': "#e6f5e9"}, 
                                 {'range': [30, 70], 'color': "#fff3cd"},
                                 {'range': [70, 100], 'color': "#f8d7da"}],
                       'bar': {'color': "#2c3e50"}}
            ))
            fig_gauge.update_layout(margin=dict(t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#333"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: #555;'>{view_title}<br>평균 실패 위험 지수</h4>", unsafe_allow_html=True)

        with col_r:
            fig_scatter = px.scatter(
                display_df, x=COL_SBA_LOAN, y='Prob', color='Prob', 
                color_continuous_scale='RdYlBu_r',
                labels={'Prob': '예측 실패 확률 (%)'} 
            )
            fig_scatter.update_layout(
                xaxis_title="대출 금액 ($)",       
                yaxis_title="예측 실패 확률 (%)",  
                xaxis_tickformat=",", margin=dict(t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: #555;'>{view_title}<br>대출 규모별 위험도 분포</h4>", unsafe_allow_html=True)
# =========================================================
# 탭 2: 실시간 심사
# =========================================================
with tab2:
    st.subheader("🔮 개별 기업 실시간 AI 정밀 심사")
    st.markdown("대출을 신청한 기업의 위치와 **상세 산업군(6자리)**을 입력하면, AI가 지역 상권의 경쟁도까지 고려하여 심사합니다.")
    st.markdown("<br>", unsafe_allow_html=True)

    # 🎨 [추가] 입력을 하나의 묶음(Form)으로 만들어서 엔터키 오류 방지
    with st.form("sim_form"):
        c1, c2, c3 = st.columns(3)
        with c1: 
            input_state = st.selectbox("📍 기업 소재지 (주)", sorted(df_main['State_Full'].dropna().unique()))
        with c2: 
            input_naics_6 = st.text_input("🏢 상세 산업군 (NAICS 6자리)", value="541110")
        with c3: 
            input_loan = st.number_input("💰 신청 대출 금액 ($)", value=100000, step=10000)
        
        # 폼 전송 버튼
        submitted = st.form_submit_button("AI 정밀 심사 실행 🚀", type="primary", use_container_width=True)

    if submitted:
        if not input_naics_6.isdigit() or len(input_naics_6) < 2:
            st.error("⚠️ 올바른 NAICS 코드를 입력해 주세요. (예: 541110)")
        else:
            naics_2_prefix = str(input_naics_6)[:2]
            exact_match = df_main[(df_main['State_Full'] == input_state) & (df_main['NAICS_2'].astype(str) == naics_2_prefix)]
            
            if not exact_match.empty:
                auto_est_val = exact_match[COL_CBP_EST].iloc[0]
            else:
                auto_est_val = df_main[df_main['NAICS_2'].astype(str) == naics_2_prefix][COL_CBP_EST].mean()
                
            if pd.isna(auto_est_val):
                auto_est_val = 0

            prob_res = model.predict_proba([[input_loan, int(input_naics_6), auto_est_val]])[0][1] * 100
            
            st.divider()
           
            # 🎨 [추가] 긴 설명을 접기/펴기 메뉴로 숨겨서 깔끔하게!
            with st.expander("ℹ️ 'AI 예측 실패 확률' 및 분석 로그 자세히 보기"):
                st.write("""
                **💡 '예측 실패 확률'이란 무엇인가요?**\n
                AI가 과거 5년간의 실제 데이터를 분석하여, **현재 신청하신 기업과 유사한 조건(업종, 대출 규모, 지역 경쟁도 등)을 가졌던 기업들이 대출금을 갚지 못하고 최종 부도(파산) 처리되었던 통계적 비율**을 뜻합니다.
                """)
                st.caption(f"*(⚙️ 시스템 로그: {input_state} 주의 {naics_2_prefix} 대분류 사업체 수인 {auto_est_val:,.0f}개를 지역 경쟁도로 자동 반영 완료)*")
            
            # 메인 심사 결과 출력
            if prob_res < 30: 
                st.success(f"### 🟢 승인 권장 (AI 위험도: {prob_res:.1f}%)")
                st.write("해당 지역의 산업 인프라와 세부 업종의 과거 패턴을 볼 때 상환 가능성이 높습니다.")
            elif prob_res < 70: 
                st.warning(f"### 🟡 조건부 승인 (AI 위험도: {prob_res:.1f}%)")
                st.write("세부 업종의 리스크가 있거나 대출 규모가 다소 큽니다. 추가 심사가 필요합니다.")
            else: 
                st.error(f"### 🔴 거절 권고 (AI 위험도: {prob_res:.1f}%)")
                st.write("과거 데이터를 볼 때, 해당 세부 업종에서 이 정도 규모의 대출은 리스크가 매우 큽니다.")
                
            st.markdown("---")
            st.subheader("📊 AI 리스크 종합 분석 리포트")
            
            # 계기판 차트
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_res, 
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "최종 AI 예측 실패 확률", 'font': {'size': 18, 'color': '#555'}},
                number = {'suffix': "%", 'font': {'size': 35, 'color': '#333'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#2c3e50", 'thickness': 0.15}, 
                    'bgcolor': "white",
                    'borderwidth': 0, # 테두리 없애서 더 깔끔하게
                    'steps': [
                        {'range': [0, 30], 'color': "#e6f5e9"}, 
                        {'range': [30, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "#e74c3c", 'width': 4},
                        'thickness': 0.75,
                        'value': prob_res
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ---------------------------------------------------------
            # 2. 🏢 상권 경쟁도(포화도)에 따른 리스크 변화 시뮬레이션
            # ---------------------------------------------------------
            st.markdown("---")
            st.subheader("🏢 상권 경쟁도 리스크 시뮬레이션")
            st.caption(f"대출 신청 금액(${input_loan:,.0f})은 고정해 두고, 해당 지역의 **동종 업계 경쟁자(사업체 수)가 변할 때** 실패 확률이 어떻게 달라지는지 AI가 가상으로 시뮬레이션합니다.")
            
            max_est = max(5000, int(auto_est_val * 2.5))
            sim_est_values = np.linspace(0, max_est, 100)
            
            sim_data = [[input_loan, int(input_naics_6), est] for est in sim_est_values]
            sim_probs = model.predict_proba(sim_data)[:, 1] * 100
            
            sim_df = pd.DataFrame({'사업체수': sim_est_values, '예측실패확률': sim_probs})
            
            fig_sim = px.line(sim_df, x='사업체수', y='예측실패확률')
            
            # 💡 [핵심 해결] opacity=0.05 였던 것을 0.15 로 올려서 색상이 선명하게 보이도록 복구했습니다!
            fig_sim.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.15, layer="below")
            fig_sim.add_hrect(y0=30, y1=70, fillcolor="orange", opacity=0.15, layer="below")
            fig_sim.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.15, layer="below")
            
            fig_sim.add_vline(x=auto_est_val, line_dash="dash", line_color="#333", 
                              annotation_text=f"현재 {input_state}주 상황<br>({auto_est_val:,.0f}개)", 
                              annotation_position="bottom right")
            
            # 💡 하얀색 도화지 배경 유지
            fig_sim.update_layout(
                yaxis_range=[0, 100], 
                xaxis_tickformat=",", 
                xaxis_title="지역 내 동종 업계 사업체 수 (경쟁도)", 
                yaxis_title="AI 예측 실패 확률 (%)",
                margin=dict(t=20, b=10)
            )
            
            st.plotly_chart(fig_sim, use_container_width=True)
            
            st.info(f"""
            💡 **AI 상권 분석 인사이트:** 현재 동종 업계 사업체가 **{auto_est_val:,.0f}개** 있는 환경에서의 리스크는 **{prob_res:.1f}%**입니다. 
            만약 그래프가 오른쪽(사업체 증가)으로 갈수록 **하락(안전해짐)**한다면, 이는 해당 산업이 인프라와 수요가 밀집된 **'핵심 상권(집적 효과)'**에 들어갈수록 생존율이 높아진다는 것을 의미합니다. 반대로 외딴곳(왼쪽)일수록 수요 부족으로 인한 리스크가 큽니다.
            """)