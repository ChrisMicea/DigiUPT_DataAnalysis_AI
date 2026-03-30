import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

# st.set_page_config(layout='wide', page_title='Titanic Explorer & Simple Model')
# button1 = st.button("test")
# print(button1)

@st.cache_data # cache-uie rezultatul functiei ca sa poata ramane in memorie si a fi tranzmis imediat
# - sa nu il incarcam de 2 sau mai multe ori
def load_seaborn_titanic():
    df = sns.load_dataset('titanic')
    # normalizeaza numele coloanelor (lowercase)
    df.columns = [c.lower() for c in df.columns]
    # titanic seaborn dataset are in coloana 'survived' valorile intregi 0/1; asigura-te ca sunt numerice
    if 'survived' in df.columns:
        df['survived'] = df['survived'].astype(int)
    return df

print('done')

# aici nu mai cache-uim pt ca se presupune ca dam upload o data sau de putine ori la aplicatie, dar
# by default, functia de mai sus se incarca si executa de fiecare data cand dam refresh sau rerender la pagina => trb sa fie rapid
def load_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    df.columns = [c.lower() for c in df.columns]
    return df

df = load_seaborn_titanic()
print('done_after')

# Sidebar: sursa datelor
st.sidebar.header('Sursă Date & Filtre')
data_source = st.sidebar.radio('Alege sursa:', ('Seaborn built-in Titanic', 'Upload CSV'))
if data_source == 'Seaborn built-in Titanic':
    df = load_seaborn_titanic()
else:
    uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
    if uploaded_file is not None:
        df = load_uploaded_csv(uploaded_file)
    else:
        st.sidebar.info('Încarcă un fișier CSV propriu sau schimbă la setul disponibil în pachetul seaborn')
        df = load_seaborn_titanic()


# Curatare elementara
def summarize_missing(dframe):
    miss = dframe.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return miss


# Filtre Sidebar (best-effort: arata doar filtrele pentru coloane des folosite)
st.sidebar.markdown('### Filtre rapide (optional)')
filterable_cols = ['pclass', 'class', 'sex', 'embarked', 'embark_town', 'who', 'adult_male', 'alone']
filters_applied = {}
for col in filterable_cols:
    if col in df.columns:
        unique_vals = df[col].dropna().unique()
        # daca sunt prea multe coloane unice, skip
        if 1 < len(unique_vals) <= 10:
            chosen = st.sidebar.multiselect(f'Filtru {col}', sorted(map(str, unique_vals)),
                                            default=sorted(map(str, unique_vals)))
            if chosen:
                # converteste coloanele initiale de tip numeric inapoi in tip numeric
                # aplica filtrele mai tarziu
                filters_applied[col] = chosen

age_min, age_max = None, None
if 'age' in df.columns:
    try:
        age_series = pd.to_numeric(df['age'], errors='coerce')
        a_min = float(np.nanmin(age_series))
        a_max = float(np.nanmax(age_series))
        age_min, age_max = st.sidebar.slider(
            'Age range', min_value=float(np.floor(a_min)),
            max_value=float(np.ceil(a_max)),
            value=(float(np.floor(a_min)), float(np.ceil(a_max)))
        )
    except Exception:
        pass

# Aplicarea filtrelor pe DataFrame
df_filtered = df.copy()
for col, chosen in filters_applied.items():
    # asigura filtrarea pe mai multe tipuri de data types
    df_filtered = df_filtered[df_filtered[col].astype(str).isin(chosen)]

if 'age' in df_filtered.columns and age_min is not None:
    df_filtered = df_filtered[df_filtered['age'].between(age_min, age_max)]


# Sectiunea principala care contine tab-urile
tabs = st.tabs(['Overview', 'EDA', 'Modeling'])
with tabs[0]:
    st.header('Overview')
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader('Data preview')
        st.write('Randuri:', df_filtered.shape[0], 'Coloane:', df_filtered.shape[1])
        st.dataframe(df_filtered.head(10))

        if st.checkbox('arata tot setul de date (poate fi foarte mare!)'):
            st.dataframe(df_filtered)

    with col2:
        st.subheader('Valori lipsă')
        miss = summarize_missing(df_filtered)
        if miss.empty:
            st.success('Nu au fost detectate valori lipsa in view-ul curent')
        else:
            st.table(miss.rename('missing_count'))

        st.subheader('Statistică descriptivă')
        numeric = df_filtered.select_dtypes(include=[np.number])
        if not numeric.empty:
            st.write(numeric.describe().T)
        else:
            st.info('No numeric columns to describe')

    st.markdown('---')
    st.subheader('Supraviețuitori')
    if 'survived' in df_filtered.columns:
        surv_counts = df_filtered['survived'].value_counts().sort_index()
        surv_pct = df_filtered['survived'].value_counts(normalize=True).sort_index()
        st.write(pd.DataFrame({'count': surv_counts, 'pct': surv_pct.round(3)}))
        fig = px.pie(values=surv_counts.values, names=['Not Survived', 'Survived'], title="Procente supraviețuitori")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Nu a fost gasită nici o coloană "survived" in setul de date.')

with tabs[1]:
    st.header('Exploratory Data Analysis (Grafice Interactive)')
    st.markdown('Pentru a vedea grafice interactive alege una dintre variabilele de mai jos.')

    left, right = st.columns(2)
    with left:
        cat_cols = df_filtered.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

        st.subheader('Categorical Plot')
        cat_col = st.selectbox(
            'Variabile tip categorice (group by)',
            options=(cat_cols if cat_cols else ['sex', 'class', 'embarked']),
            index=0 if cat_cols else 0
        )
        if 'survived' in df_filtered.columns:
            agg = df_filtered.groupby([cat_col, 'survived']).size().reset_index(name='count')
            fig_cat = px.bar(
                agg, x=cat_col, y='count', color='survived', barmode='group',
                labels={'survived': 'survived'}
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            fig_cat = px.histogram(df_filtered, x=cat_col)
            st.plotly_chart(fig_cat, use_container_width=True)

        st.subheader('Scatter / Perechi numerice')
        xcol = st.selectbox('X', options=(num_cols if num_cols else ['age', 'fare']), index=0 if num_cols else 0)
        ycol = st.selectbox('Y', options=(num_cols if num_cols else ['fare', 'age']), index=0 if num_cols else 0)
        color_by = st.selectbox('Colr by (optional)', options=(cat_cols + ['survived']) if cat_cols else ['survived'], index=0)
        fig_scatter = px.scatter(df_filtered, x=xcol, y=ycol, color=color_by, hover_data=df_filtered.columns)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right:
        st.subheader('Distributii')
        dist_col = st.selectbox(
            'Alege o variabilă pentru a vedea distribuția ei',
            options=(num_cols if num_cols else ['age', 'fare'])
        )
        fig_hist = px.histogram(
            df_filtered, x=dist_col, nbins=30,
            color='survived' if 'survived' in df_filtered.columns else None
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader('Grafice tip heatmap')
        cat1 = st.selectbox('Category 1', options=(cat_cols if cat_cols else ['class', 'sex']), index=0)
        cat2 = st.selectbox('Category 2', options=(cat_cols if cat_cols else ['sex', 'embarked']), index=1)
        if 'survived' in df_filtered.columns:
            pivot = pd.pivot_table(df_filtered, index=cat1, columns=cat2, values='survived', aggfunc=np.mean)
            fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index.astype(str), colorscale='Viridis'))
            fig_heat.update_layout(title='Rata medie de supraviețuire', xaxis_title=cat2, yaxis_title=cat1)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info('Adauga o coloana cu numele "survived" pentru a obtine un grafic tip heatmap.')

with tabs[2]:
    st.header('Modelare -- Regresia Logistica')
    st.markdown(
        """
    Această secțiune antrenează un pipeline simplu de Regresie Logistică. \n
    Selectează variabilele țintă (implicit se selectează ‘survived’), proporția pentru impartirea test/train și pornește antrenarea modelului
        """
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        # Selectia variabilelor in UI
        available_cols = df_filtered.columns.tolist()
        default_feats = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        features = st.multiselect('Features (predictori)', options=available_cols,
                                  default=[c for c in default_feats if c in available_cols])
    with col_b:
        target = st.selectbox('Variabila tintă pe care modelul sa o prezică', options=available_cols,
                              index=available_cols.index('survived') if 'survived' in available_cols else 0)
        test_size = st.slider('Setează parametrul: Test size', min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        random_state = st.number_input('Setează parametrul: Random state', value=42, step=1)
    
    if not features:
        st.warning('Selecteaza cel putin o variabila care sa fie prezisă.')
    else:
        if st.button('Antrenează modelul'):
            data = df_filtered[features + [target]].copy()
            # Elimina randurile pentru care variabila tinta e nula
            data = data.dropna(subset=[target])
            # Elimina randurile cand toate variabilele lipsesc
            data = data.dropna(axis=0, how='all', subset=features)

            # Determina tipurile variabilelor
            X = data[features]
            y = data[target].astype(int) if pd.api.types.is_integer_dtype(data[target]) or pd.api.types.is_bool_dtype(
                data[target]) else pd.factorize(data[target])[0]

            # Îimărțirea train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Tipurile coloanelor pentru pipeline
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = [c for c in X.columns if c not in numeric_features]

            # Pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('onehot', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ], remainder='drop'
            )

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('clf', LogisticRegression(max_iter=1000))
                # ('model', LinearRegression())
            ])

            with st.spinner('Modelul se antrenează...', show_time=True):
                model.fit(X_train, y_train)

            # Predictori & metrici
            y_pred = model.predict(X_test)
            y_proba = None
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                pass

            acc = accuracy_score(y_test, y_pred)
            st.success(f'Accuracy on test set: {acc:.3f}')

            st.subheader('Classification report')
            report = classification_report(y_test, y_pred, output_dict=True)
            st.table(pd.DataFrame(report).transpose())

            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(
                data=go.Heatmap(z=cm, x=['pred_0', 'pred_1'][:cm.shape[1]], y=['True_0', 'true_1'][:cm.shape[0]],
                                colorscale='Blues', showscale=True, text=cm, texttemplate='%{text}'))
            fig_cm.update_layout(title='Confusion matrix', xaxis_title='Predicted', yaxis_title='Actual')
            st.plotly_chart(fig_cm, use_container_width=True)

            if y_proba is not None and len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
                fig_roc.update_layout(title='ROC curve', xaxis_title='False Positive Rate',
                                      yaxis_title='True Positive Rate')
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info('curba ROC nu e disponibila (date incomplete)')