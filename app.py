import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. Global Variables & Settings ---
THRESHOLD_LOW = 0.15   # Must be under 15% for 🟢 Safe (Strict standard)
THRESHOLD_HIGH = 0.35  # Unconditionally 🔴 High Risk if 35% or more (Target recall 80% defense line)

COL_SBA_NAICS = 'NAICSCode'
COL_SBA_LOAN  = 'GrossApproval'
COL_SBA_TERM  = 'TerminMonths'
COL_CBP_EST   = 'Number of establishments (ESTAB)'
COL_CBP_EMP   = 'Number of employees (EMP)'

# US Census Bureau Official NAICS 2-digit Sector Mapping
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

# Overall Page Configuration
st.set_page_config(page_title="SBA-CBP Intelligent Evaluation System", page_icon="🏦", layout="wide")

# --- 2. Load Data & Model ---
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
    st.error("🚨 Model (`sba_model.pkl`) or data (`sba_app_data_lite.csv`) file could not be found.")
    st.stop()


# =========================================================
# 3. Main UI (One-page Dashboard Configuration)
# =========================================================
st.title("🏦 SBA-CBP Intelligent Loan Risk Diagnosis System")
st.markdown("Diagnoses the **probability of loan approval** by analyzing the average scale of the selected region and industry sector in real-time.")
st.write("") 

all_codes = get_naics_2digit_codes(df_main)

# Group input section into a clean container
with st.container(border=True):
    st.markdown("#### 📝 Enter Loan Evaluation Conditions")
    c1, c2, c3, c4 = st.columns(4)
    with c1: in_st = st.selectbox("📍 Business Location (State)", sorted(df_main['State_Full'].dropna().unique()), key="in_st")
    with c2: in_na = st.selectbox("🏢 Industry Sector", options=all_codes, format_func=format_naics_display, key="in_na")
    with c3: 
        in_loan = st.number_input(
            "💰 Requested Loan Amount ($)", 
            min_value=10000, max_value=5000000, value=100000, step=10000, 
            help="The statutory maximum limit for the US SBA 7(a) loan program is $5,000,000."
        )
    with c4: in_term = st.number_input("⏳ Desired Repayment Term (Months)", value=60, step=12, min_value=12)
    
    st.write("")
    btn_calc = st.button("Start AI Precision Diagnosis & Simulation 🚀", type="primary", use_container_width=True)

if btn_calc:
    n2 = str(in_na).zfill(2)
    ref = df_main[(df_main['State_Full'] == in_st) & (df_main['NAICS_2'].astype(str).str.zfill(2) == n2)]
    
    if ref.empty:
        est = df_main[COL_CBP_EST].mean()
        emp = df_main[COL_CBP_EMP].mean()
        st.info("⚠️ Due to a lack of detailed data for this region/industry, calculations are based on the national average scale.")
    else:
        est = ref[COL_CBP_EST].iloc[0]
        emp = ref[COL_CBP_EMP].iloc[0]
        
    avg_emp = emp / max(est, 1)
    sector_name = NAICS_SECTOR_MAPPING.get(n2, "")
    
    # Strict score conversion formula to achieve 80% target recall
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
        
        # [Business Rule] Immediate reject if monthly repayment per employee exceeds $2,679 (Proxy DSR)
        if monthly_burden_per_emp > 2679:
            is_knockout = True
            knockout_reason = f"Failed to meet repayment capacity (DSR) standards (Estimated monthly repayment \${monthly_burden_per_emp:,.0f} / Exceeds regulatory limit of \$2,679)"
            raw_s = 5  
            raw_p = 99.9
            
        return raw_p, raw_s, is_knockout, knockout_reason

    curr_p, curr_s, is_knockout, knockout_reason = get_p_and_s(in_loan, in_term)
    
    st.divider()
    
    # Top Summary Indicators
    st.markdown(f"##### 📊 Benchmark Indicators: {in_st} Region / {sector_name}")
    st.caption(f"The average number of employees per establishment in this industry is analyzed to be **{avg_emp:.1f}**.")
    
    res_col1, res_col2 = st.columns([1, 1.5], gap="large")
    
    with res_col1:
        fig_sc = go.Figure(go.Indicator(
            mode="gauge+number", value=curr_s, 
            title={'text': "Current Condition Safety Score", 'font': {'size': 20}},
            number={'suffix': " pts", 'font': {'size': 40}},
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
                f"**🚨 [Notice of Automatic Loan Rejection]** \n\n"
                f"The conditions you requested exceed our internal risk management standards, making it impossible to calculate a loan limit.\n\n"
                f"**▪ Reason for Rejection:** {knockout_reason} \n\n"
                f"**▪ Recommendation:** Please significantly lower the requested loan amount or extend the repayment period and try again.",
            )
        else:
            if curr_s >= 70: 
                status_color = "🟢"
                status_text = "Safe (High probability of approval)"
            elif curr_s >= 50: 
                status_color = "🟡"
                status_text = "Caution (Condition adjustment recommended)"
            else: 
                status_color = "🔴"
                status_text = "Risk (High probability of rejection)"
            
            avg_loan_for_industry = ref[COL_SBA_LOAN].mean() if not ref.empty else df_main[COL_SBA_LOAN].mean()
            avg_term_for_industry = ref[COL_SBA_TERM].mean() if not ref.empty else df_main[COL_SBA_TERM].mean()
            
            industry_base_p, _, _, _ = get_p_and_s(avg_loan_for_industry, avg_term_for_industry)
            relative_risk = curr_p / max(industry_base_p, 0.1)
            
            # Using HTML block to manage spacing tightly and emphasize warnings
            st.markdown(
                f"### Diagnosis Status: {status_color} {status_text}\n"
                f"<div style='margin-top: -10px; margin-bottom: 15px; line-height: 1.6;'>"
                f"<span style='font-size: 1.05em;'>Current Estimated Default Risk: <b>{curr_p:.1f}%</b></span><br>"
                f"<span style='color: #0056b3; font-weight: 500;'>👉 This is {relative_risk:.1f} times higher than the industry average risk ({industry_base_p:.1f}%).</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # --- Binary Search Algorithm ---
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

        st.write("") 
        
        # Suggestion Cards
        if curr_s >= 70 and not is_knockout:
            st.success("✨ **The current conditions are sufficiently safe based on the standards for this region and industry.**", icon="✅")
        else:
            if not is_knockout:
                st.warning("⚠️ **Loan approval may be difficult under current conditions.** Consider the options below to raise the safety score to 70 or higher.", icon="💡")
            else:
                st.markdown("#### 💡 AI Suggested Conditions for Approval (70+ points)")
                
            opt1, opt2 = st.columns(2)
            with opt1:
                with st.container(border=True):
                    st.markdown("📉 **Option A. Reduce Loan Amount**")
                    if safe_loan_max > 0: 
                        st.markdown(f"While keeping the repayment term ({in_term} months),<br>lower the loan amount to **${safe_loan_max:,.0f}** or less.", unsafe_allow_html=True)
                    else: 
                        st.markdown(f"Due to the **high base risk or small scale** of this industry, automatic approval (70 pts) is difficult even if the loan amount is minimized.", unsafe_allow_html=True)
            with opt2:
                with st.container(border=True):
                    st.markdown("⏳ **Option B. Extend Repayment Term**")
                    if safe_term_min > 0: 
                        st.markdown(f"While keeping the loan amount (${in_loan:,.0f}),<br>extend the repayment term to **{safe_term_min} months** or more.", unsafe_allow_html=True)
                    else: 
                        st.markdown(f"The requested amount is too large compared to the industry scale, making approval difficult even with the maximum term extension.", unsafe_allow_html=True)

    # --- Simulation Graphs ---
    st.markdown("---")
    st.markdown("#### 📈 Safety Score Simulation Based on Condition Changes")
    st.caption("Adjust the conditions so the graph line enters the green (70 pts) zone. (The flat 5-point area represents automatic rejection due to exceeding regulatory limits.)")
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        loan_sim_range = np.linspace(max(10000, in_loan * 0.2), min(in_loan * 1.5, 5000000), 30)
        scores_by_loan = [get_p_and_s(l, in_term)[1] for l in loan_sim_range]
        
        fig_loan = go.Figure()
        fig_loan.add_trace(go.Scatter(x=loan_sim_range, y=scores_by_loan, mode='lines', name='Safety Score', line=dict(color='#007bff', width=3)))
        fig_loan.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Safety Baseline (70 pts)")
        fig_loan.add_vline(x=in_loan, line_dash="solid", line_color="#dc3545", annotation_text="Current Req. Amt")
        fig_loan.update_layout(title=f"Loan Limit with Fixed {in_term}-Month Term", xaxis_title="Loan Amount ($)", yaxis_title="Safety Score", yaxis=dict(range=[0, 100]), margin=dict(t=40, b=40))
        st.plotly_chart(fig_loan, use_container_width=True)

    with sim_col2:
        term_sim_range = np.arange(12, max(240, in_term + 60), 12)
        scores_by_term = [get_p_and_s(in_loan, t)[1] for t in term_sim_range]
        
        fig_term = go.Figure()
        fig_term.add_trace(go.Scatter(x=term_sim_range, y=scores_by_term, mode='lines', name='Safety Score', line=dict(color='#fd7e14', width=3)))
        fig_term.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Safety Baseline (70 pts)")
        fig_term.add_vline(x=in_term, line_dash="solid", line_color="#dc3545", annotation_text="Current Req. Term")
        fig_term.update_layout(title=f"Required Term with Fixed ${in_loan:,.0f} Loan", xaxis_title="Repayment Term (Months)", yaxis_title="Safety Score", yaxis=dict(range=[0, 100]), margin=dict(t=40, b=40))
        st.plotly_chart(fig_term, use_container_width=True)


