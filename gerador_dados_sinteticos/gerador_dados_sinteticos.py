# %% [markdown]
# # Atividade: Gerando Dados Sintéticos Realistas
# 
# ## Objetivo
# Gerar um dataset sintético simulando logs de acesso a um e-commerce, definindo variáveis, distribuições e avaliando a qualidade dos dados.

# %% [markdown]
# ### Passo 1: Definição das Variáveis e Dicionário de Dados (2 pontos)
# 
# Crie um dicionário para as variáveis do seu DataFrame. O DataFrame deve incluir variáveis contínuas, categóricas e com valores nulos.  
# O dicionário de dados deve incluir 'nome da variável', 'tipo' e 'descrição'.
# 
# | Variável          | Tipo      | Descrição                              | Distribuição Sugerida          |
# |-------------------|-----------|----------------------------------------|--------------------------------|
# | session_id        | String    | ID único da sessão                     | UUID ou sequência numérica     |
# | user_id           | Integer   | ID do usuário                          | Sequência numérica - 10% não logados (Null)   |
# | timestamp         | Datetime  | Timestamp do evento                    | Distribuição temporal diária (Poisson)  |
# | page_url          | String    | URL da página acessada                 | Padrão `/produto/{id}`         |
# | session_duration  | Float     | Duração da sessão em segundos          | Exponencial                    |
# | add_to_cart       | Boolean   | Indicador de adicionar ao carrinho     | Bernoulli(p=0.2)               |
# | purchase          | Boolean   | Indicador de compra                    | Bernoulli(p=0.05)              |

# %% [markdown]
# ### Passo 2: Geração do Dataset em Python Básico (3 pontos)
# 
# Gere o dataset utilizando apenas bibliotecas como `numpy`, `pandas` e `datetime` (sem bibliotecas como `Faker`).
# 
# **Desafios extras**:  
# - Adicionar dependências entre variáveis (ex: `session_duration` maior para `add_to_cart`).  
# - Gerar URLs realistas (ex: `/produto/{id}` com `id` sequencial).  
# - Considerar períodos de alta sazonalidade como natal, dia das mães e Black Friday.

# %% [markdown]
# #### 2.1 Instalação das dependências

# %%
import subprocess
import sys

# Build the command: [python, -m, pip, install, packages...]
cmd = [
    sys.executable,  # ensures you use the current Python interpreter
    "-m", "pip",
    "install",
    "numpy",
    "pandas",
    "matplotlib"
]

# Run it, raise an exception if it fails
result = subprocess.run(
    cmd,
    capture_output=True,  # so you can inspect stdout/stderr
    text=True,            # decode output to str
    check=True            # will raise CalledProcessError on non-zero exit
)

print("🟢 Installed successfully!")
print(result.stdout)


# %% [markdown]
# #### 2.2 Importando bibliotecas

# %%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# %% [markdown]
# #### 2.3 Definindo o período e gerando datas com sazonalidade

# %% [markdown]
# O período contemplado será de todos os dias do ano de 2024

# %%
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
days = pd.date_range(start_date, end_date, freq='D')

# %% [markdown]
# #### 2.4 Gerando timestamps de sessões diárias com variação sazonal

# %% [markdown]
# Aqui será gerado os numéros de sessões por dia com variação no Natal e na black friday, a fim de gerar os ruídos similares aos encontrados em datasets reais. No natal haverá 30% mais visitas, já na black friday as visitas dobrarão
# 

# %%
sessions = []
base_sessions = 300

for day in days:
    n = np.random.poisson(base_sessions)
    # Natal
    if day.month == 12 and 20 <= day.day <= 24:
        n = int(n * 1.3)

    # Black Friday
    if day.month == 11 and day.weekday() == 4 and day.day >= 23:
        n = int(n * 2)
    for _ in range(n):
        random_time = day + timedelta(
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        sessions.append(random_time)

timestamps = sessions

# %% [markdown]
# #### 2.5 Construindo DataFrame inicial com timestamp e IDs

# %% [markdown]
# Inicia as outras colunas do dataframe, como session_id, user_id, product_id e page_url

# %%
df = pd.DataFrame({'timestamp': timestamps})
df['session_id'] = ['sess_' + str(i) for i in range(len(df))]
df['user_id'] = np.random.randint(1, 10000, size=len(df))
df['product_id'] = np.random.randint(1, 1000, size=len(df))
df['page_url'] = df['product_id'].apply(lambda x: f"/produto/{x}")

# %% [markdown]
# #### 2.6 Adicionando variáveis de comportamento

# %% [markdown]
# Adiciona a coluna `add_to_cart` com 20% de probabilidade de ser verdadeira

# %%
# add_to_cart com probabilidade 0.2
df['add_to_cart'] = np.random.rand(len(df)) < 0.2

# %% [markdown]
# ##### Ajustando session_duration conforme `add_to_cart`

# %% [markdown]
# Gera o tempo de sessão e aumenta em 50% o tempo de sessão em que `add_to_cart` é `true`

# %%
# Duração exponencial
df['session_duration'] = np.random.exponential(scale=300, size=len(df))
# Aumenta duração em sessões com add_to_cart
df.loc[df['add_to_cart'], 'session_duration'] *= 1.5

# %% [markdown]
# ##### Definindo `purchase` dependente de `add_to_cart`

# %% [markdown]
# Adiciona a coluna `purchase`. Ela tem 20% de chance de ser verdadeira se a coluna `add_to_cart` é verdadeira

# %%
purchase_prob = np.where(df['add_to_cart'], 0.2, 0.0)
df['purchase'] = np.random.rand(len(df)) < purchase_prob

# %% [markdown]
# ##### Inserindo valores nulos

# %% [markdown]
# Insere valor nulo em `user_id` para representar usuários não logados. A probabilidade do usuário ser nulo é de 10%

# %%
mask = np.random.rand(len(df)) < 0.1
df.loc[mask, 'user_id'] = pd.NA
df.isnull().sum()

# %% [markdown]
# ##### Versão final do Dataset

# %%
df.head()

# %% [markdown]
# Tamanho do DF

# %%
len(df)

# %% [markdown]
# ### Passo 3: Avaliação da Qualidade dos Dados (3 pontos)
# 
# **Dica**  
# - Capriche na visualização dos dados (i.e., use os gráficos certos para demonstrar o seu ponto)

# %% [markdown]
# #### 3.1 Estatísticas e valores ausentes

# %%
# Percentual de valores ausentes
missing = df.isnull().mean() * 100
print(missing)

# Estatísticas descritivas
df.describe()

# %% [markdown]
# #### 3.2 Visualizações

# %% [markdown]
# ##### Histograma da duração da sessão

# %%
# Histograma de duração de sessão
plt.figure(figsize=(8,4))
plt.hist(df['session_duration'], bins=50)
plt.title('Histograma de Duração de Sessão')
plt.xlabel('Duração (segundos)')
plt.ylabel('Frequência')
plt.show()

# %% [markdown]
# A duração da sessão é representada por uma distribuição poisson

# %% [markdown]
# ##### Relação entre a quantidade de `add_to_cart` vs `purchase`

# %%
# Calculando os totais
totals = df[['add_to_cart', 'purchase']].sum()

# Criando o gráfico
plt.figure(figsize=(8,5))
totals.plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title('Total de Ações: Adicionar ao Carrinho vs Compra')
plt.ylabel('Quantidade')
plt.xticks(rotation=0)
plt.show()

# %% [markdown]
# Dado que `purchase` só ocorre se `add_to_cart` for verdadeiro, faz sentido a correlação demonstrada

# %% [markdown]
# ##### Quantidade de sessões por dia

# %%
daily_counts = df.set_index('timestamp').resample('D').size()
plt.figure(figsize=(10,4))
plt.plot(daily_counts.index, daily_counts.values)
plt.title('Número de Sessões por Dia')
plt.xlabel('Data')
plt.ylabel('Sessões')
plt.show()

# %% [markdown]
# A partir disso, é possível ver o aumento realizado entre o número de sessões na black friday e no natal

# %% [markdown]
# #### 3.3 Boxplot da duração de sessão por add_to_cart
# Este gráfico compara a distribuição do tempo de sessão entre sessões que resultaram em adição ao carrinho e as que não resultaram, permitindo visualizar a dispersão e possíveis outliers.

# %%
# Boxplot de duration por add_to_cart
plt.figure(figsize=(8,4))
df.boxplot(column='session_duration', by='add_to_cart')
plt.title('Duração de Sessão vs Add to Cart')
plt.suptitle('')
plt.xlabel('Adicionou ao Carrinho')
plt.ylabel('Duração (segundos)')
plt.show()

# %% [markdown]
# A partir desse gráfico, fica visível a quantidade de outliers não esperados. Dado que a duração da sessão não foi criada a partir de uma distribuição normal, há uma grande quantidade de outliers. Entretanto, é possível observar o aumento provocado do tempo de sessão quando há a adição do produto no carrinho.

# %% [markdown]
# #### 3.4 Taxa de conversão mensal
# Calcula e plota a proporção de sessões que resultaram em compra por mês, sinalizando tendências sazonais na conversão.

# %%
# Taxa de conversão mensal (purchase rate)
monthly_rate = df.set_index('timestamp').resample('ME').purchase.mean()
plt.figure(figsize=(10,4))
plt.plot(monthly_rate.index, monthly_rate.values, marker='o')
plt.title('Taxa de Conversão Mensal')
plt.xlabel('Mês')
plt.ylabel('Proporção de Compras')
plt.show()

# %% [markdown]
# Com essa visualização é possível observar que a falta da relação entre compras nos períodos da black friday e do natal gerou uma inconsistência entre a relação do numéro de sessões e de compras.

# %% [markdown]
# #### 3.5 Top 10 produtos mais visitados
# Identifica os 10 produtos com maior número de acessos, auxiliando na análise de popularidade de itens.

# %%
# Top 10 produtos por número de acessos
top10 = df['product_id'].value_counts().nlargest(10)
plt.figure(figsize=(8,4))
plt.bar(top10.index.astype(str), top10.values)
plt.title('Top 10 Produtos por Acessos')
plt.xlabel('ID do Produto')
plt.ylabel('Número de Acessos')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# Como não houve a criação da relação entre itens e compras, a distribuição se manteve constante ao longo do top10, tendo poucas variações.

# %% [markdown]
# ### Passo 4: Considerações Finais (2 pontos)
# *Pontos positivos do DataFrame*
# - Há a inserção de sazonalidade nos dias do Natal e Black Friday
# - Há uma relação direta entre `add_to_cart` e `purchase`, de modo a trazer coerência aos dados, já que não há como realizar uma compra sem itens no carrinho
# - Há a inserção de valores nulos para representar usuários não logados
# - As distribuições escolhidas foram escolhidas para se parecer com a distribuição de dataframes reais
# 
# 
# *Limitações do DataFrame*
# - Não modela variações de comportamento por dia da semana (ex.: finais de semana vs dias úteis).
# - Cada linha corresponde a um clique isolado, sem agrupar múltiplas páginas numa mesma sessão.
# - A sazonalidade é simplificada e não leva em conta promoções específicas ou campanhas de marketing.
# - Não há sequência lógica de navegação por produto ou recomendações; IDs são independentes e aleatórios.
# 


