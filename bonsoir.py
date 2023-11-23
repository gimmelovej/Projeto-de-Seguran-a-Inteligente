import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest 
import folium
from folium.plugins import MarkerCluster

def filter_and_normalize(df):
    # Implemente a filtragem e normalização dos dados conforme necessário
    # Exemplo: converter endereços IP para formato padrão, normalizar datas, etc.
    return df

def calculate_risk_score(port):
    # Implemente a lógica para calcular o escore de risco com base na porta
    # Aqui, estou usando uma lógica simples, você pode ajustar conforme necessário
    if port == 22:
        return 8
    elif port == 80:
        return 5
    elif port == 443:
        return 7
    else:
        return 3

def classify_vulnerabilities(df):
    # Verificar se as colunas necessárias estão presentes
    if "Vulnerability" not in df.columns or "Risk Score" not in df.columns:
        print("Colunas necessárias para classificação de vulnerabilidades não estão presentes.")
        return None

    # Codificar as vulnerabilidades
    le = LabelEncoder()
    df['Vulnerability_Label'] = le.fit_transform(df['Vulnerability'])

    # Dividir o conjunto de dados em recursos (X) e rótulos (y)
    X = df[['Risk Score']]
    y = df['Vulnerability_Label']

    # Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar e treinar o modelo de Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Avaliar a precisão do modelo
    accuracy = model.score(X_test, y_test)
    print(f'A precisão do modelo de classificação de vulnerabilidades é: {accuracy:.2f}')

    # Retornar informações adicionais
    return {
        'accuracy': accuracy,
        'feature_importances': model.feature_importances_,
        'vulnerability_labels': le.classes_,
        'confusion_matrix': pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
    }

def generate_security_report(df, classification_info):
    # Inicializar a variável report
    report = ""

    # Análise Temporal Mensal
    df["Month"] = df["Timestamp"].dt.to_period("M")
    monthly_counts = df.groupby("Month").size()

    # Aplicar teste estatístico para verificar a estacionaridade
    p_value = adfuller(monthly_counts)[1]

    # Gráficos da Análise Temporal
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    monthly_counts.plot(kind="bar", color="skyblue")
    plt.title("Atividades de Segurança ao Longo do Tempo")
    plt.xlabel("Mês")
    plt.ylabel("Contagem de Eventos")

    plt.subplot(3, 1, 2)
    monthly_counts.plot(color="orange", marker="o")
    plt.title("Tendência Temporal")
    plt.xlabel("Mês")
    plt.ylabel("Tendência")

    plt.tight_layout()
    plt.savefig("security_temporal_analysis.png")

    # ... (restante do código da análise temporal)

    # Análise de Agrupamento (Clustering) de IPs
    le = LabelEncoder()
    df["IP Label"] = le.fit_transform(df["IP Address"])

    kmeans = KMeans(n_clusters=3, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[["IP Label"]])

    # Gráfico de Agrupamento
    plt.figure(figsize=(10, 6))
    plt.scatter(df["IP Label"], df["Vulnerability"], c=df["Cluster"], cmap="viridis")
    plt.title("Agrupamento de IPs")
    plt.xlabel("IP Label")
    plt.ylabel("Vulnerability")
    plt.colorbar()
    plt.savefig("ip_cluster_analysis.png")

    # ... (restante do código do agrupamento de IPs)

    # Análise de Risco por IP
    risk_by_ip = df.groupby("IP Address")["Risk Score"].mean().sort_values(ascending=False)

    # Gráfico de Barras do Risco por IP
    plt.figure(figsize=(12, 6))
    risk_by_ip.plot(kind="bar", color="salmon")
    plt.title("Risco Médio por IP")
    plt.xlabel("IP Address")
    plt.ylabel("Risco Médio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("risk_by_ip.png")

    # ... (restante do código da análise de risco por IP)

    # Distribuição das Vulnerabilidades
    vulnerability_distribution = df["Vulnerability"].value_counts()

    # Gráfico de Barras da Distribuição das Vulnerabilidades
    plt.figure(figsize=(12, 6))
    vulnerability_distribution.plot(kind="bar", color="skyblue")
    plt.title("Distribuição das Vulnerabilidades")
    plt.xlabel("Vulnerabilidade")
    plt.ylabel("Contagem")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("vulnerability_distribution.png")

    # ... (restante do código da distribuição das vulnerabilidades)

    # Detecção de Anomalias nas Vulnerabilidades
    if "Vulnerability" in df.columns:
        le = LabelEncoder()
        df['Vulnerability_Label'] = le.fit_transform(df['Vulnerability'])

        isolation_forest_vuln = IsolationForest(contamination=0.05)  # Ajuste conforme necessário
        df['Anomalia_Vulnerability'] = isolation_forest_vuln.fit_predict(df[['Vulnerability_Label']])

        plt.figure(figsize=(10, 6))
        plt.scatter(df.index, df['Vulnerability_Label'], c=df['Anomalia_Vulnerability'], cmap='viridis', label='Eventos')
        plt.title('Detecção de Anomalias nas Vulnerabilidades')
        plt.xlabel('Índice do Evento')
        plt.ylabel('Vulnerability Label')
        plt.colorbar()
        plt.savefig("anomaly_detection_vulnerability.png")


    # Adicionar informações da classificação ao relatório
    report += f"""
    **Classificação de Vulnerabilidades:**
    - Precisão do Modelo: {classification_info['accuracy']:.2f}
    
    **Importância das Características:**
    {pd.DataFrame({'Feature': ['Risk Score'], 'Importance': classification_info['feature_importances']})}
    
    **Rótulos de Vulnerabilidades:**
    {classification_info['vulnerability_labels']}
    
    **Matriz de Confusão:**
    {classification_info['confusion_matrix']}
    """

    # Adicionar informações da Análise de Risco por IP ao relatório
    report += f"""
    **Análise de Risco por IP:**
    - Top 5 IPs com Maior Risco Médio:
    {risk_by_ip.head()}

    ![Gráfico de Risco por IP](risk_by_ip.png)
    """

    # Adicionar informações da Distribuição das Vulnerabilidades ao relatório
    report += f"""
    **Distribuição das Vulnerabilidades:**
    {vulnerability_distribution}

    ![Gráfico de Distribuição das Vulnerabilidades](vulnerability_distribution.png)
    """

    # Adicionar informações da Detecção de Anomalias nas Vulnerabilidades ao relatório
    if "Vulnerability" in df.columns:
        report += f"""
        **Detecção de Anomalias nas Vulnerabilidades:**
        ![Gráfico de Anomalias nas Vulnerabilidades](anomaly_detection_vulnerability.png)
        """


    # Salvar o relatório em um arquivo
    with open("security_report.md", "w") as report_file:
        report_file.write(report)


def analyze_security_data(csv_filename):
    # Carregar dados do CSV
    df = pd.read_csv(csv_filename, parse_dates=["Timestamp"], dayfirst=True)
    
    # Filtragem e Normalização
    df = filter_and_normalize(df)

    # Análise Temporal Mensal
    df["Month"] = df["Timestamp"].dt.to_period("M")
    monthly_counts = df.groupby("Month").size()

    # Aplicar teste estatístico para verificar a estacionaridade
    p_value = adfuller(monthly_counts)[1]

    # Gerar gráficos
    plt.figure(figsize=(14, 8))

    # Gráfico de Atividades ao Longo do Tempo
    plt.subplot(3, 1, 1)
    monthly_counts.plot(kind="bar", color="skyblue")
    plt.title("Atividades de Segurança ao Longo do Tempo")
    plt.xlabel("Mês")
    plt.ylabel("Contagem de Eventos")

    # Gráfico de Tendência
    plt.subplot(3, 1, 2)
    monthly_counts.plot(color="orange", marker="o")
    plt.title("Tendência Temporal")
    plt.xlabel("Mês")
    plt.ylabel("Tendência")

    plt.tight_layout()
    plt.savefig("security_temporal_analysis.png")

    # Análise básica
    total_entries = len(df)
    unique_ips = df["IP Address"].nunique()
    vulnerability_counts = df["Vulnerability"].value_counts()

    # Análise por Categoria de Vulnerabilidade
    category_counts = df["Vulnerability Category"].value_counts()

    # Análise por Tipo de Vulnerabilidade
    vulnerability_counts = df["Vulnerability"].value_counts()

    # Análise por Localização
    geo_counts = df['Latitude'].astype(str) + '_' + df['Longitude'].astype(str)

    # Gerar gráfico de barras
    plt.figure(figsize=(10, 6))
    vulnerability_counts.plot(kind="bar", color="skyblue")
    plt.title("Contagem de Tipos de Vulnerabilidades")
    plt.xlabel("Tipo de Vulnerabilidade")
    plt.ylabel("Contagem")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("vulnerability_chart.png")

    # Análise de Agrupamento (Clustering) de IPs
    le = LabelEncoder()
    df["IP Label"] = le.fit_transform(df["IP Address"])

    kmeans = KMeans(n_clusters=3, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[["IP Label"]])

    # Gerar gráfico de agrupamento
    plt.figure(figsize=(10, 6))
    plt.scatter(df["IP Label"], df["Vulnerability"], c=df["Cluster"], cmap="viridis")
    plt.title("Agrupamento de IPs")
    plt.xlabel("IP Label")
    plt.ylabel("Vulnerability")
    plt.colorbar()
    plt.savefig("ip_cluster_analysis.png")

    # Análise Intermediária de Risco de Portas
    df["Risk Score"] = df["Port"].apply(calculate_risk_score)

    # Gerar gráfico de barras do risco por porta
    plt.figure(figsize=(10, 6))
    df.groupby("Port")["Risk Score"].mean().sort_values(ascending=False).plot(kind="bar", color="salmon")
    plt.title("Risco Médio por Porta")
    plt.xlabel("Porta")
    plt.ylabel("Risco Médio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("risk_by_port.png")

    # Análise Geoespacial
    if "Latitude" in df.columns and "Longitude" in df.columns:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]))

        # Criar um mapa interativo com agrupamento de marcadores
        m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=10)
        marker_cluster = MarkerCluster().add_to(m)

        for index, row in gdf.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"IP: {row['IP Address']}\nVulnerability: {row['Vulnerability']}",
            ).add_to(marker_cluster)

        m.save("security_map.html")
    
    # Detecção de Anomalias no Risco da Porta
    if "Port" in df.columns:
        # Utilizar Isolation Forest para detecção de anomalias no risco da porta
        isolation_forest_port = IsolationForest(contamination=0.05)  # Ajuste conforme necessário
        df['Anomalia_Port'] = isolation_forest_port.fit_predict(df[['Risk Score']])

        # Visualizar eventos normais e anomalias
        plt.figure(figsize=(10, 6))
        plt.scatter(df.index, df['Risk Score'], c=df['Anomalia_Port'], cmap='viridis')
        plt.title('Detecção de Anomalias no Risco da Porta')
        plt.xlabel('Índice do Evento')
        plt.ylabel('Risk Score')
        plt.colorbar()
        plt.savefig("anomaly_detection_port.png")

    # Classificação de Vulnerabilidades
    classification_info = classify_vulnerabilities(df)

    # Gerar Relatório
    generate_security_report(df, classification_info)

if __name__ == "__main__":
    analyze_security_data("security_data.csv")