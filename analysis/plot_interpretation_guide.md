# 🧠 Model Dynamics Plot Interpretation Guide

## 📊 Understanding Vector Flow Plots

The plots you've generated show **vector flow fields** that represent how each model's internal states change over time during decision-making. Here's how to interpret them:

### 🎯 **What Are Vector Flow Plots?**

Each arrow in the plot represents:
- **Starting Point** (arrow base): Current internal state of the model
- **Direction & Length** (arrow): How the state changes in the next time step
- **Flow Patterns**: Overall learning/decision dynamics

---

## 🔬 **Model-Specific Interpretations**

### 1. **LSTM Dynamics** (`lstm_dynamics_by_diagnosis.png`)
- **Axes**: Hidden State Dimensions 1 & 2
- **What to look for**:
  - **Convergence points**: Where arrows point toward (stable states)
  - **Spirals**: Learning patterns - tight spirals = fast adaptation
  - **Spread**: How varied the internal representations are

**Clinical Interpretation**:
- **Healthy**: Should show stable, organized flow patterns
- **Depression**: May show more chaotic or stuck patterns (less exploration)
- **Bipolar**: Could show more extreme or variable flow patterns

### 2. **SPICE Dynamics** (`spice_dynamics_by_diagnosis.png`)
- **Axes**: Symbolic State Dimensions 1 & 2
- **What to look for**:
  - **Discrete clusters**: SPICE uses symbolic representations
  - **Transitions**: How the model moves between symbolic states
  - **Interpretable patterns**: SPICE is designed to be interpretable

**Clinical Interpretation**:
- Different diagnosis groups should show distinct symbolic transition patterns
- More organized patterns may indicate better cognitive control

### 3. **RNN Dynamics** (`rnn_dynamics_by_diagnosis.png`)
- **Axes**: Hidden State Dimensions 1 & 2
- **What to look for**:
  - **Recurrent patterns**: Loops and cycles in the flow
  - **Memory effects**: How past decisions influence current states
  - **Temporal dynamics**: Evolution over time

**Clinical Interpretation**:
- Healthy individuals may show more stable recurrent patterns
- Clinical groups might show disrupted temporal dynamics

### 4. **GQL Dynamics** (`gql_dynamics_by_diagnosis.png`)
- **Axes**: Q-Value for Action 0 vs Q-Value for Action 1
- **What to look for**:
  - **Diagonal patterns**: Q-values updating based on rewards
  - **Convergence**: How quickly Q-values stabilize
  - **Exploration vs Exploitation**: Spread vs concentration of points

**Clinical Interpretation**:
- **Healthy**: Balanced exploration-exploitation trade-off
- **Depression**: May show reduced exploration (stuck in one quadrant)
- **Bipolar**: Could show extreme swings between high/low Q-values

---

## 🎨 **Visual Pattern Recognition**

### 🟢 **Healthy Patterns (Expected)**
- **Smooth, organized flows**
- **Balanced exploration** of state space
- **Convergent patterns** (arrows pointing toward stable regions)
- **Moderate spread** of starting points

### 🔴 **Depression Patterns (Potential)**
- **Reduced exploration** (arrows concentrated in small regions)
- **Stuck dynamics** (very short arrows, little change)
- **Bias toward negative states** (if applicable to model)
- **Less variability** in flow patterns

### 🔵 **Bipolar Patterns (Potential)**
- **Extreme dynamics** (very long arrows, big changes)
- **Highly variable patterns** (inconsistent flow directions)
- **Switching between regions** (arrows pointing to distant areas)
- **Unstable convergence** (no clear stable points)

---

## 📈 **Individual Dynamics Interpretation**

The individual participant plots show:
- **Participant-specific patterns**: How each person's model behaves
- **Consistency within diagnosis**: Do people with same diagnosis show similar patterns?
- **Individual differences**: Unique patterns within diagnostic groups

### 🔍 **What to Compare**:
1. **Within-participant consistency**: Are the arrows generally pointing in similar directions?
2. **Between-participant differences**: How do the three participants (Healthy, Depression, Bipolar) differ?
3. **Model differences**: How does the same participant look across different models?

---

## 🚨 **Key Questions to Ask**

### 📊 **Diagnosis Comparison**:
1. **Do clinical groups show different flow patterns than healthy controls?**
2. **Are depression and bipolar patterns distinct from each other?**
3. **Which models show the clearest diagnostic differences?**

### 🧪 **Model Comparison**:
1. **Which model captures the most clinically meaningful patterns?**
2. **Do simpler models (GQL) show similar patterns to complex ones (LSTM)?**
3. **Which model shows the most interpretable results?**

### 👥 **Individual Differences**:
1. **How much variability is there within each diagnostic group?**
2. **Are there participants who don't fit their diagnostic group pattern?**
3. **Do any participants show similar patterns across models?**

---

## 💡 **Clinical Significance**

### 🎯 **What These Patterns Mean**:
- **Computational markers**: Different flow patterns could be biomarkers for mental health
- **Treatment targets**: Understanding disrupted dynamics could guide interventions
- **Personalized medicine**: Individual patterns could inform personalized treatments

### 📝 **Reporting Results**:
When describing your results, focus on:
1. **Qualitative differences** between groups
2. **Consistency across models** (do multiple models show similar patterns?)
3. **Clinical relevance** (how do patterns relate to known symptoms?)
4. **Model interpretability** (which models provide the clearest insights?)

---

## 🔧 **Technical Notes**

- **Arrow density**: Subsampled for visualization (max 200 arrows per plot)
- **Axis ranges**: Automatically scaled to data range
- **Color coding**: Consistent across diagnosis groups
- **State extraction**: Based on model's internal representations during actual participant trials

---

## 📚 **Next Steps for Analysis**

1. **Quantitative measures**: Calculate flow metrics (convergence speed, exploration area, stability)
2. **Statistical testing**: Test if group differences are significant
3. **Correlation analysis**: Relate flow patterns to behavioral measures
4. **Predictive modeling**: Use flow patterns to predict diagnosis or outcomes
