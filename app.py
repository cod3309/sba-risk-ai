import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. 전역 변수 및 설정 ---
THRESHOLD_LOW = 0.15   # 15% 미만이어야 🟢 안전 (깐깐한 기준)
THRESHOLD_HIGH = 0.35  # 35% 이상이면 무조건 🔴 위험 (타겟 재현율 80% 방어선)

COL_SBA_NAICS = 'NAICSCode'
COL_SBA_LOAN  = 'GrossApproval'
COL_SBA_TERM  = 'TerminMonths'
COL_CBP_EST   = 'Number of establishments (ESTAB)'
COL_CBP_EMP   = 'Number of employees (EMP)'

# 미국 인구조사국 공식 NAICS 2자리 대분류 매핑
NAICS_SECTOR_MAPPING = {
    '11': 'Agriculture, Forestry, Fishing and Hunting',
    '21': 'Mining, Quarrying, and Oil and Gas Extraction',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale Trade',
    '44': 'Retail Trade',
    '45': 'Retail Trade',
    '48': 'Transportation and Warehousing',
    '49': 'Transportation and Warehousing',
    '51': 'Information',
    '52': 'Finance and Insurance',
    '53': 'Real Estate and Rental and Leasing',
    '54': 'Professional, Scientific, and Technical Services',
    '55': 'Management of Companies and Enterprises',
    '56': 'Administrative and Support and Waste Management and Remediation Services',
    '61': 'Educational Services',
    '62': 'Health Care and Social Assistance',
    '71': 'Arts, Entertainment, and Recreation',
    '72': 'Accommodation and Food Services',
    '81': 'Other Services (except Public Administration)',
    '92': 'Public Administration'
}

def format_naics_display(code):
    code_str = str(code).zfill(2)[:2]
    sector_name = NAICS_SECTOR_MAPPING.get(code_str, "Unknown Industry")
    return f"{code_str} - {sector_name}"

# 전체 페이지 여백 및 탭 설정
st.set_page_config(page_title="SBA-CBP 지능형 심사 시스템", page_icon="🏦", layout="wide")

# --- 2. 데이터 및 모델 로드 ---
@st.cache_resource
def load_model(): 
    return joblib.load('sba_model.pkl')

@st.cache_data
def load_data(): 
    return pd.read_csv('sba_app_data_lite.csv')

@st.cache_data
def get_naics_2digit_codes(df):
    codes = df['NAICS_2'].dropna().astype(str).str.zfill(2).unique().tolist()
    return sorted(codes)

try:
    model = load_model()
    df_main = load_data()
except Exception as e:
    st.error("🚨 모델(`sba_model.pkl`) 또는 데이터(`sba_app_data_lite.csv`) 파일을 찾을 수 없습니다.")
    st.stop()


# =========================================================
# 3. 메인 UI (원페이지 구성)
# =========================================================
st.title("🏦 SBA-CBP 지능형 대출 리스크 진단 시스템")
st.markdown("해당 지역 및 산업군(Sector)의 평균 체급을 실시간으로 분석하여, **대출 심사 승인 가능성을 진단**합니다.")
st.write("") # 시각적 여백

all_codes = get_naics_2digit_codes(df_main)

# 💡 [디자인 개선] 입력 섹션을 하나의 깔끔한 박스로 그룹화
with st.container(border=True):
    st.markdown("#### 📝 대출 심사 조건 입력")
    c1, c2, c3, c4 = st.columns(4)
    with c1: in_st = st.selectbox("📍 사업장 위치", sorted(df_main['State_Full'].dropna().unique()), key="in_st")
    with c2: in_na = st.selectbox("🏢 산업군(Sector)", options=all_codes, format_func=format_naics_display, key="in_na")
    with c3: 
        in_loan = st.number_input(
            "💰 대출 희망액 ($)", 
            min_value=10000, max_value=5000000, value=100000, step=10000, 
            help="미국 SBA 7(a) 대출 프로그램의 법정 최대 한도는 $5,000,000 입니다."
        )
    with c4: in_term = st.number_input("⏳ 희망 상환기간 (개월)", value=60, step=12, min_value=12)
    
    st.write("")
    btn_calc = st.button("AI 정밀 진단 및 시뮬레이션 시작 🚀", type="primary", use_container_width=True)

if btn_calc:
    n2 = str(in_na).zfill(2)
    ref = df_main[(df_main['State_Full'] == in_st) & (df_main['NAICS_2'].astype(str).str.zfill(2) == n2)]
    
    if ref.empty:
        est = df_main[COL_CBP_EST].mean()
        emp = df_main[COL_CBP_EMP].mean()
        st.info("⚠️ 해당 지역/산업군의 세부 데이터가 부족하여 전국 평균 체급을 기준으로 산출합니다.")
    else:
        est = ref[COL_CBP_EST].iloc[0]
        emp = ref[COL_CBP_EMP].iloc[0]
        
    avg_emp = emp / max(est, 1)
    sector_name = NAICS_SECTOR_MAPPING.get(n2, "")
    
    # 💡 [핵심 수정] 타겟 재현율 80%를 달성하기 위한 깐깐한 점수 변환 공식
    def calculate_smooth_score(prob):
        if prob <= 15:
            return 100 - (prob * 2) 
        elif prob <= 35:
            return 70 - (prob - 15) 
        else:
            return max(5, 50 - ((prob - 35) * 1.5))

    def get_p_and_s(l, t):
        l_r = l / max(avg_emp, 1)
        m_b = l / max(t, 1)
        
        input_df = pd.DataFrame([{
            'State_Full': in_st, 'NAICS_2': n2, COL_SBA_LOAN: l, COL_CBP_EST: est,
            COL_SBA_TERM: t, 'Avg_EMP_per_EST': avg_emp, 'Loan_to_Avg_EMP_Ratio': l_r, 'Monthly_Burden': m_b
        }])
        
        input_df['State_Full'] = input_df['State_Full'].astype('category')
        input_df['NAICS_2'] = input_df['NAICS_2'].astype('category')
        
        raw_p = model.predict_proba(input_df)[0][1] * 100
        raw_s = calculate_smooth_score(raw_p)
        
        is_knockout = False
        knockout_reason = ""
        
        monthly_burden_per_emp = m_b / max(avg_emp, 1)
        
        # [비즈니스 룰] 1인당 월 상환액이 $2,679 초과 시 즉시 거절
        if monthly_burden_per_emp > 2679:
            is_knockout = True
            knockout_reason = f"상환 여력(DSR) 기준 미달 (추정 월 상환액 \${monthly_burden_per_emp:,.0f} / 규정 한도 \$2,679 초과)"
            raw_s = 5  
            raw_p = 99.9
            
        return raw_p, raw_s, is_knockout, knockout_reason

    curr_p, curr_s, is_knockout, knockout_reason = get_p_and_s(in_loan, in_term)
    
    st.divider()
    
    # 상단 요약 지표 (대시보드 느낌 강화)
    st.markdown(f"##### 📊 기준 지표: {in_st} 지역 / {sector_name}")
    st.caption(f"이 산업군의 사업체당 평균 직원 수는 **{avg_emp:.1f}명**으로 분석됩니다.")
    
    res_col1, res_col2 = st.columns([1, 1.5], gap="large")
    
    with res_col1:
        fig_sc = go.Figure(go.Indicator(
            mode="gauge+number", value=curr_s, 
            title={'text': "현재 조건 안전 점수", 'font': {'size': 20}},
            number={'suffix': "점", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100]}, 'bar': {'color': "#2c3e50"},
                'steps': [{'range': [0, 50], 'color': "#f8d7da"}, 
                          {'range': [50, 70], 'color': "#fff3cd"}, 
                          {'range': [70, 100], 'color': "#d4edda"}]}
        ))
        fig_sc.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_sc, use_container_width=True)

    with res_col2:
        if is_knockout:
            st.error(
                f"**🚨 [여신 심사 자동 부결 안내]** \n\n"
                f"고객님께서 요청하신 조건은 내부 리스크 관리 기준을 초과하여 대출 한도 산출이 불가합니다.\n\n"
                f"**▪ 부결 사유:** {knockout_reason} \n\n"
                f"**▪ 권고 사항:** 대출 신청 금액을 대폭 하향 조정하시거나, 상환 기간을 연장하여 재조회해 주시기 바랍니다.",
                
            )
        else:
            if curr_s >= 70: 
                status_color = "🟢"
                status_text = "안전 (승인 가능성 높음)"
            elif curr_s >= 50: 
                status_color = "🟡"
                status_text = "주의 (조건 조정 권장)"
            else: 
                status_color = "🔴"
                status_text = "위험 (반려 가능성 높음)"
            
            avg_loan_for_industry = ref[COL_SBA_LOAN].mean() if not ref.empty else df_main[COL_SBA_LOAN].mean()
            avg_term_for_industry = ref[COL_SBA_TERM].mean() if not ref.empty else df_main[COL_SBA_TERM].mean()
            
            industry_base_p, _, _, _ = get_p_and_s(avg_loan_for_industry, avg_term_for_industry)
            relative_risk = curr_p / max(industry_base_p, 0.1)
            
            st.markdown(f"### 진단 상태: {status_color} {status_text}")
            st.write(f"현재 예상 부도 위험도: **{curr_p:.1f}%**")
            st.caption(f"👉 동종 업계 평균 위험도({industry_base_p:.1f}%) 대비 **{relative_risk:.1f}배** 수준입니다.")

        # --- 이진 탐색 알고리즘 ---
        low_loan, high_loan, safe_loan_max = 10000, 5000000, 0
        while low_loan <= high_loan:
            mid_loan = (low_loan + high_loan) // 2
            mid_loan = (mid_loan // 1000) * 1000 
            if get_p_and_s(mid_loan, in_term)[1] >= 70:
                safe_loan_max = mid_loan
                low_loan = mid_loan + 1000  
            else:
                high_loan = mid_loan - 1000 

        low_term, high_term, safe_term_min = 12, 360, 0
        while low_term <= high_term:
            mid_term = (low_term + high_term) // 2
            if get_p_and_s(in_loan, mid_term)[1] >= 70:
                safe_term_min = mid_term
                high_term = mid_term - 1  
            else:
                low_term = mid_term + 1   
                
        if safe_term_min > 0 and get_p_and_s(in_loan, safe_term_min)[1] < 70:
            safe_term_min = 0 

        st.write("") # 간격 띄우기
        
        # 💡 [디자인 개선] 대안 제시 부분을 시각적으로 눈에 띄는 카드로 구성
        if curr_s >= 70 and not is_knockout:
            st.success("✨ **현재 조건이 해당 지역 및 산업군 기준에 비추어 볼 때 충분히 안전한 수준입니다.**", icon="✅")
        else:
            if not is_knockout:
                st.warning("⚠️ **현재 조건으로는 대출 승인이 어려울 수 있습니다.** 안전 점수를 70점 이상으로 높이려면 아래 조건을 고려해 보세요.", icon="💡")
            else:
                st.markdown("#### 💡 AI가 제안하는 승인 가능(70점 이상) 조건")
                
            opt1, opt2 = st.columns(2)
            with opt1:
                with st.container(border=True):
                    st.markdown("📉 **옵션 A. 대출금 축소**")
                    if safe_loan_max > 0: 
                        st.markdown(f"상환 기간({in_term}개월) 유지 시,<br>대출금을 **${safe_loan_max:,.0f}** 이하로 낮추세요.", unsafe_allow_html=True)
                    else: 
                        st.markdown(f"해당 산업의 **기본 리스크가 높거나 체급이 작아**, 대출액을 최소로 줄여도 자동 승인(70점)이 어렵습니다.", unsafe_allow_html=True)
            with opt2:
                with st.container(border=True):
                    st.markdown("⏳ **옵션 B. 상환기간 연장**")
                    if safe_term_min > 0: 
                        st.markdown(f"대출 금액(${in_loan:,.0f}) 유지 시,<br>상환 기간을 **{safe_term_min}개월** 이상으로 늘리세요.", unsafe_allow_html=True)
                    else: 
                        st.markdown(f"요청 금액이 동종 업계 체급 대비 너무 커서 기간을 최대로 늘려도 승인이 어렵습니다.", unsafe_allow_html=True)

    # --- 시뮬레이션 그래프 ---
    st.markdown("---")
    st.markdown("#### 📈 조건 변화에 따른 안전 점수 시뮬레이션")
    st.caption("그래프의 선이 초록색(70점) 구간에 들어오도록 조건을 조정해야 합니다. (점수가 5점인 바닥 구간은 규정 한도 초과에 의한 자동 거절 구간입니다.)")
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        loan_sim_range = np.linspace(max(10000, in_loan * 0.2), min(in_loan * 1.5, 5000000), 30)
        scores_by_loan = [get_p_and_s(l, in_term)[1] for l in loan_sim_range]
        
        fig_loan = go.Figure()
        fig_loan.add_trace(go.Scatter(x=loan_sim_range, y=scores_by_loan, mode='lines', name='안전 점수', line=dict(color='#007bff', width=3)))
        fig_loan.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="안전 기준 (70점)")
        fig_loan.add_vline(x=in_loan, line_dash="solid", line_color="#dc3545", annotation_text="현재 신청액")
        fig_loan.update_layout(title=f"상환기간 {in_term}개월 고정 시, 대출액 한도", xaxis_title="대출 금액 ($)", yaxis_title="안전 점수", yaxis=dict(range=[0, 100]), margin=dict(t=40, b=40))
        st.plotly_chart(fig_loan, use_container_width=True)

    with sim_col2:
        term_sim_range = np.arange(12, max(240, in_term + 60), 12)
        scores_by_term = [get_p_and_s(in_loan, t)[1] for t in term_sim_range]
        
        fig_term = go.Figure()
        fig_term.add_trace(go.Scatter(x=term_sim_range, y=scores_by_term, mode='lines', name='안전 점수', line=dict(color='#fd7e14', width=3)))
        fig_term.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="안전 기준 (70점)")
        fig_term.add_vline(x=in_term, line_dash="solid", line_color="#dc3545", annotation_text="현재 신청기간")
        fig_term.update_layout(title=f"대출액 ${in_loan:,.0f} 고정 시, 필요 상환기간", xaxis_title="상환 기간 (개월)", yaxis_title="안전 점수", yaxis=dict(range=[0, 100]), margin=dict(t=40, b=40))
        st.plotly_chart(fig_term, use_container_width=True)