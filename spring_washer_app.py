""""
Script to estimate force, stiffness and stress for
spring-washers of any dimensions or materials.

Reference: https://www.engineersedge.com/belleville_spring.htm
"""

import numpy as np
import altair as alt
import pandas as pd
import streamlit as st


def force(xx):
    global t, f1, f2
    xx = xx/t
    return f1*xx*((f2 - xx) * (f2 - xx/2) + 1)
    # return (f2 - xx) * (f2 - xx/2) + 1


def stiffness(xx):
    global t, f1, f2
    xx = xx/t
    st1 = f1*((f2 - xx) * (f2 - xx / 2) + 1)
    st2 = f1*xx*(-(f2 - xx/2) - (f2 - xx)/2)
    return (st1 + st2)/t


def stress(xx):
    global t, s1, f2, beta, gamma
    xx = xx/t
    return s1*xx*(beta*(f2-xx/2) + gamma)


def main():
    global E, t, nu, D, d, H
    global beta, gamma, f1, f2, s1
    # Constants ===============================================================
    h = H - t       # Height of truncated cone [m]
    delta = D/d     # Diameter ratio [-]

    alpha = ((delta - 1)/delta)**2 / \
            ((delta + 1)/(delta - 1) -
             2/np.log(delta)) / np.pi   # Auxiliary variable [-]

    beta = (6*((delta-1)/np.log(delta) - 1)
            / np.log(delta))/np.pi      # Auxiliary variable [-]

    gamma = (3*(delta-1) /
             np.pi)/np.log(delta)       # Auxiliary variable [-]

    # Force terms --------------------------------------------------
    f1 = 4*E*t**4/((1-nu**2)*alpha*D**2)
    f2 = h/t

    # Stress terms -------------------------------------------------
    s1 = 4*E*t**2/((1-nu**2)*alpha*D**2)

    # Calculations ============================================================
    x = np.linspace(0, h, 200)      # Deflection vector [m]
    F = force(x)
    k = stiffness(x)
    sigma = stress(x)
    df = pd.DataFrame({'Deflection (m)': x,
                       'Force (N)': F,
                       'Stiffness (N/m)': k,
                       'Stress (Pa)': sigma
                       })

    # Plots ===================================================================
    fig1 = alt.Chart(df).mark_line().encode(
           alt.X('Deflection (m):Q'), alt.Y('Force (N):Q').scale(zero=False)).interactive()
    fig2 = alt.Chart(df).mark_line().encode(
           alt.X('Deflection (m):Q'), alt.Y('Stiffness (N/m)').scale(zero=False)).interactive()
    fig3 = alt.Chart(df).mark_line().encode(
           alt.X('Deflection (m):Q'), alt.Y('Stress (Pa)').scale(zero=False)).interactive()

    return [fig1, fig2, fig3]


# Streamlit ===================================================================
# Title and main text -------------------------------------
st.title('Spring washer mechanical properties')
st.text('This is a program to calculate the force, the stiffness,'
        ' and the stresses\nin a spring washer as a function'
        ' of its geometry, material and deflection.')
st.image('spring_washer_dimensions.jpg',
         caption='Reference for dimensions used below.')

# Inputs --------------------------------------------------
st.header('Inputs')
E = st.number_input('Elasticity modulus [GPa]', min_value=0.01, max_value=1e5,
                    value=210.0, step=10.0)*1e9
t = st.number_input('Washer material thickness [mm]', min_value=0.01, max_value=5.0,
                    value=.9, step=0.1)*1e-3
nu = st.number_input('Poisson\'s ratio [-]', min_value=0.01, max_value=3.0,
                     value=0.29, step=0.01)
D = st.number_input('Outer diameter [mm]', min_value=1.0, max_value=100.0,
                    value=16.0, step=0.01)*1e-3
d = st.number_input('Inner diameter [mm]', min_value=0.01, max_value=D*1e3-0.01,
                    value=8.2, step=0.01)*1e-3
H = st.number_input('Washer total height [mm]', min_value=t+0.01, max_value=10.0,
                    value=1.25, step=0.01)*1e-3

# Figure --------------------------------------------------
figs = main()
tabs = st.tabs(['Force', 'Stiffness', 'Stress'])
for i in range(3):
    with tabs[i]:
        st.altair_chart(figs[i], theme='streamlit', use_container_width=True)
