"""Streamlit application for the EV Adoption Likelihood Classification System.

Phase 4 of the project.  Reuses ``utils.predict`` for all inference —
does NOT load pickle files directly.
"""

import streamlit as st
import pandas as pd

from utils.predict import predict, model_information

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EV Adoption Classifier",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem 1.8rem 1.8rem 1.8rem;
    margin-bottom: 1.5rem;
}
.section-card h3 {
    margin-top: 0;
    margin-bottom: 0.25rem;
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
}
.section-card .subtitle {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 1.25rem;
}
.section-divider {
    margin: 0.5rem 0 1.25rem 0;
    border: 0;
    height: 1px;
    background: linear-gradient(to right, #e5e7eb, transparent);
}
.result-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
.result-card .pred-class {
    font-size: 2rem;
    font-weight: 700;
    color: #166534;
    line-height: 1.2;
}
.result-card .pred-label {
    font-size: 0.85rem;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.button-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 0.5rem;
}
div[data-testid="stForm"] {
    border: none;
    padding: 0;
}
div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] + div[data-testid="element-container"] {
    margin-top: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "error_message" not in st.session_state:
    st.session_state.error_message = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _build_sidebar() -> None:
    """Populate the sidebar with project overview, model info, and usage tips."""

    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/car.png",
            width=64,
        )
        st.markdown("## EV Adoption Classifier")

        with st.expander("Project Overview", expanded=True):
            st.markdown(
                "Predict an individual's likelihood of adopting an electric "
                "vehicle based on demographic, behavioural, and infrastructural "
                "features.  Built with a **K-Nearest Neighbors** classifier."
            )

        with st.expander("Algorithm Used"):
            try:
                info = model_information()
                st.markdown(
                    f"**{info['algorithm']}**\n\n"
                    f"Input features: {info['num_features']}\n\n"
                    f"Target classes: {', '.join(info['classes'])}"
                )
            except Exception:
                st.warning("Model information unavailable.  Ensure model files exist.")

        with st.expander("Dataset Summary"):
            st.markdown(
                "The model was trained on **50,000 records** from the **Global "
                "EV Adoption Behavior (2026)** dataset covering 23 attributes "
                "across demographics, commuting patterns, charging "
                "infrastructure, and attitudinal scores."
            )

        with st.expander("Developer Information"):
            st.markdown(
                "Built as a capstone machine learning project.\n\n"
                "Technologies: Python, scikit-learn, pandas, Streamlit."
            )

        with st.expander("Model Statistics"):
            try:
                info = model_information()
                st.markdown(
                    f"- **Algorithm:** KNN (k={info['algorithm'].split('k=')[1].split(',')[0]})\n"
                    f"- **Features:** {info['num_features']}\n"
                    f"- **Classes:** {', '.join(info['classes'])}\n"
                    f"- **Model file:** `{info['model_file']}`"
                )
            except Exception:
                st.warning("Statistics unavailable.")

        with st.expander("Instructions"):
            st.markdown(
                "1. Fill in all fields in the form.\n"
                "2. Click **Predict EV Adoption Likelihood**.\n"
                "3. View predicted class, confidence, and probability breakdown.\n"
                "4. Use **Reset** to clear and start over."
            )

        st.markdown("---")
        st.caption("EV Adoption Likelihood Classification System \u00B7 2026")


# ---------------------------------------------------------------------------
# Form widgets
# ---------------------------------------------------------------------------


def _numeric(label: str, key: str, **kwargs) -> float:
    """Render a ``st.number_input`` and return its value."""
    return st.number_input(label, key=key, **kwargs)


def _slider(label: str, key: str, **kwargs) -> float:
    """Render a ``st.slider`` and return its value."""
    return st.slider(label, key=key, **kwargs)


def _dropdown(label: str, key: str, options, **kwargs):
    """Render a ``st.selectbox`` and return its value."""
    return st.selectbox(label, options=options, key=key, **kwargs)


# ---------------------------------------------------------------------------
# Input gathering
# ---------------------------------------------------------------------------


def _gather_inputs() -> dict:
    """Collect all widget values from session state.

    Keys match the expected feature columns so the map is trivial.
    """
    feature_names = [
        "age", "annual_income", "daily_commute_km", "weekly_travel_distance_km",
        "vehicle_age_years", "education_level", "fuel_expense_per_month",
        "charging_station_accessibility", "nearest_charging_station_km",
        "electricity_cost_per_kwh", "monthly_energy_consumption_kwh",
        "monthly_charging_cost", "home_charging_available",
        "environmental_awareness_score", "government_incentive_awareness",
        "technology_affinity_score", "range_anxiety_score",
        "battery_replacement_concern", "ev_knowledge_score", "city_type",
        "current_vehicle_type", "previous_ev_experience",
    ]
    return {name: st.session_state.get(name) for name in feature_names}


# ---------------------------------------------------------------------------
# Form sections
# ---------------------------------------------------------------------------


def _section_personal() -> None:
    """Personal Information section."""
    st.markdown(
        '<div class="section-card">'
        "<h3>Personal Information</h3>"
        '<p class="subtitle">Demographic details that influence EV adoption propensity.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        _numeric(
            "Age",
            "age",
            min_value=18,
            max_value=80,
            value=35,
            step=1,
            help="Age of the individual (years).",
        )
        _dropdown(
            "Education Level",
            "education_level",
            options=["High School", "Bachelor", "Master", "PhD"],
            help="Highest education level attained.",
        )
    with c2:
        _numeric(
            "Annual Income ($)",
            "annual_income",
            min_value=0,
            max_value=300_000,
            value=50_000,
            step=1_000,
            format="%d",
            help="Total annual income in USD.",
        )
        _dropdown(
            "City Type",
            "city_type",
            options=["Urban", "Suburban", "Rural"],
            help="Type of residential area.",
        )


def _section_vehicle() -> None:
    """Vehicle & Travel section."""
    st.markdown(
        "</div>"
        '<div class="section-card">'
        "<h3>Vehicle & Travel</h3>"
        '<p class="subtitle">Current vehicle details and commuting patterns.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        _dropdown(
            "Current Vehicle Type",
            "current_vehicle_type",
            options=["Hatchback", "Sedan", "SUV", "Truck"],
            help="Type of vehicle currently owned.",
        )
        _numeric(
            "Daily Commute (km)",
            "daily_commute_km",
            min_value=0.0,
            max_value=150.0,
            value=25.0,
            step=0.5,
            help="One-way daily commuting distance in kilometres.",
        )
    with c2:
        _numeric(
            "Vehicle Age (years)",
            "vehicle_age_years",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
            help="Age of the current primary vehicle.",
        )
        _numeric(
            "Weekly Travel Distance (km)",
            "weekly_travel_distance_km",
            min_value=0.0,
            max_value=1000.0,
            value=200.0,
            step=5.0,
            help="Total distance travelled per week.",
        )


def _section_charging() -> None:
    """Charging & Cost section."""
    st.markdown(
        "</div>"
        '<div class="section-card">'
        "<h3>Charging & Cost</h3>"
        '<p class="subtitle">Fuel, electricity, and charging infrastructure details.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        _numeric(
            "Fuel Expense / Month ($)",
            "fuel_expense_per_month",
            min_value=0.0,
            max_value=1000.0,
            value=250.0,
            step=10.0,
            help="Monthly fuel expenditure in USD.",
        )
        _slider(
            "Charging Station Accessibility",
            "charging_station_accessibility",
            min_value=1,
            max_value=10,
            value=5,
            help="How accessible are charging stations? (1 = very difficult, 10 = very easy).",
        )
        _numeric(
            "Monthly Energy Consumption (kWh)",
            "monthly_energy_consumption_kwh",
            min_value=0.0,
            max_value=1000.0,
            value=150.0,
            step=5.0,
            help="Total household monthly energy consumption.",
        )
    with c2:
        _numeric(
            "Nearest Charging Station (km)",
            "nearest_charging_station_km",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Distance to the nearest public charging station.",
        )
        _numeric(
            "Electricity Cost / kWh ($)",
            "electricity_cost_per_kwh",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.01,
            format="%.2f",
            help="Cost of electricity per kilowatt-hour in USD.",
        )
        _numeric(
            "Monthly Charging Cost ($)",
            "monthly_charging_cost",
            min_value=0.0,
            max_value=300.0,
            value=30.0,
            step=5.0,
            help="Monthly cost of charging an EV at home.",
        )
    _dropdown(
        "Home Charging Available",
        "home_charging_available",
        options=[0, 1],
        format_func={0: "No", 1: "Yes"}.get,
        help="Does the individual have access to home charging?",
    )


def _section_behaviour() -> None:
    """Behaviour & Awareness section."""
    st.markdown(
        "</div>"
        '<div class="section-card">'
        "<h3>Behaviour & Awareness</h3>"
        '<p class="subtitle">Attitudinal scores and prior EV experience (scale 1–10).</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        _slider(
            "Environmental Awareness",
            "environmental_awareness_score",
            min_value=1,
            max_value=10,
            value=5,
            help="Level of environmental concern (1 = low, 10 = high).",
        )
        _slider(
            "Technology Affinity",
            "technology_affinity_score",
            min_value=1,
            max_value=10,
            value=5,
            help="Familiarity / openness to new technology.",
        )
        _slider(
            "Range Anxiety",
            "range_anxiety_score",
            min_value=1,
            max_value=10,
            value=5,
            help="Worry about EV driving range (1 = no worry, 10 = very anxious).",
        )
    with c2:
        _slider(
            "Govt Incentive Awareness",
            "government_incentive_awareness",
            min_value=1,
            max_value=10,
            value=5,
            help="Awareness of government EV incentives.",
        )
        _slider(
            "EV Knowledge",
            "ev_knowledge_score",
            min_value=1,
            max_value=10,
            value=5,
            help="Self-assessed knowledge about electric vehicles.",
        )
        _slider(
            "Battery Replacement Concern",
            "battery_replacement_concern",
            min_value=1,
            max_value=10,
            value=5,
            help="Concern about EV battery replacement cost.",
        )
    _dropdown(
        "Previous EV Experience",
        "previous_ev_experience",
        options=[0, 1],
        format_func={0: "No", 1: "Yes"}.get,
        help="Has the individual previously owned or driven an EV?",
    )


# ---------------------------------------------------------------------------
# Form builder
# ---------------------------------------------------------------------------


def _build_form() -> None:
    """Render the prediction form in clearly separated section cards."""
    _section_personal()
    _section_vehicle()
    _section_charging()
    _section_behaviour()


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------


def _display_result(result: dict) -> None:
    """Render prediction result with metrics, progress bars, and a bar chart.

    Parameters
    ----------
    result : dict
        Output from ``predict()`` with keys ``class``, ``confidence``,
        ``probabilities``.
    """
    st.markdown(
        '<div class="result-card">', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="pred-label">Prediction Result</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="pred-class">{}</p>'.format(result["class"]),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Confidence", "{:.2%}".format(result["confidence"]))

    with c2:
        st.caption("Probability Distribution")

    probs = result["probabilities"]
    max_prob = max(probs.values()) if max(probs.values()) > 0 else 1.0
    for label, prob in probs.items():
        pct = prob * 100
        st.markdown(
            "**{}**: {:.1f}%".format(label, pct),
            help="Probability that the individual belongs to the '{}' class.".format(
                label
            ),
        )
        st.progress(prob / max_prob)

    prob_df = pd.DataFrame(
        {
            "Class": list(probs.keys()),
            "Probability": [round(v * 100, 1) for v in probs.values()],
        }
    )
    st.bar_chart(prob_df, x="Class", y="Probability", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.success("Prediction complete!")


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def _reset() -> None:
    """Clear the stored prediction result and rerun."""
    st.session_state.prediction_result = None
    st.session_state.error_message = None
    st.rerun()


# ---------------------------------------------------------------------------
# About Model
# ---------------------------------------------------------------------------


def _show_model_info() -> None:
    """Display model metadata in an expandable section."""
    with st.expander("About the Model"):
        try:
            info = model_information()
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Algorithm**\n\n{}".format(info["algorithm"]))
                st.markdown(
                    "**Number of features**\n\n{}".format(info["num_features"])
                )
            with col_b:
                st.markdown(
                    "**Target classes**\n\n{}".format(", ".join(info["classes"]))
                )
                st.markdown("**Model file**\n\n`{}`".format(info["model_file"]))
            st.markdown("**Expected features**")
            st.write(info["expected_features"])
        except Exception as exc:
            st.warning("Could not load model information: {}".format(exc))


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------


def _build_footer() -> None:
    """Render the footer separator and copyright."""
    st.markdown("---")
    st.caption(
        "EV Adoption Likelihood Classification System  |  "
        "Built with Streamlit & scikit-learn  |  2026"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    _build_sidebar()

    st.title("EV Adoption Likelihood Classifier")
    st.markdown(
        "Predict whether an individual has a **High**, **Medium**, or **Low** "
        "likelihood of adopting an electric vehicle based on their demographic "
        "profile, commuting patterns, charging infrastructure access, and "
        "attitudinal scores."
    )

    # --- Form ---
    with st.form("prediction_form", clear_on_submit=False):
        _build_form()
        st.markdown(
            '</div><div class="button-row">',
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button(
            "Predict EV Adoption Likelihood",
            type="primary",
            use_container_width=True,
        )

    # --- Reset ---
    c_left, c_btn, c_right = st.columns([3, 1, 3])
    with c_btn:
        st.button("Reset", on_click=_reset, use_container_width=True)

    # --- Prediction ---
    if submitted:
        with st.spinner("Running prediction..."):
            try:
                data = _gather_inputs()
                result = predict(data)
                st.session_state.prediction_result = result
                st.session_state.error_message = None
            except (ValueError, TypeError, RuntimeError, FileNotFoundError) as exc:
                st.session_state.prediction_result = None
                st.session_state.error_message = str(exc)

    # --- Error display ---
    if st.session_state.error_message:
        st.error(
            "{}\n\nPlease review your inputs and try again.".format(
                st.session_state.error_message
            )
        )

    # --- Result display ---
    if st.session_state.prediction_result is not None:
        _display_result(st.session_state.prediction_result)

    # --- Model info & footer ---
    _show_model_info()
    _build_footer()


if __name__ == "__main__":
    main()
