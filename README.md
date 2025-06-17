# ENVIE APENAS A PASTA `enviar_prova`

# Objetivo
O objetivo desse Repo é gerar várias referências para possibilitar a montagem do [notebook para a prova](./enviar_prova/prova.ipynb).

# Exemplos
Os exemplos contém as ponderadas realizadas e avaliadas previamente como nota 10. O objetivo da existência desses arquivos é gerar insights valiosos sobre como é possível abordar requisitos de negócio ao longo da prova prática - o que pode ser um ir além.

Ademais, eles também contém a geração de dados a partir de diferentes distribuições, como `gamma`, `normal` e `beta`.

- Exemplo de geração de dados sintéticos: [Geração de Dados Sintéticos](./gerador_dados_sinteticos/gerando_dados_sinteticos.ipynb)

- Exemplo de simulador teste AB: [Simulador teste AB](./simulador_teste_ab/simulador_teste_ab.ipynb)

- Exemplo de matriz markov e Monte Carlo: [Prova passada](./prova-ano-passado.ipynb)

# ChartGenerator

> Importante: para usar o ChartGenerator, você deve ter instalado o `matplotlib` e o `pandas` no seu arquivo jupyter (`ipynb`)

```python
%pip install numpy pandas matplotlib
```

O arquivo `ChartGenerator.py` define a classe `ChartGenerator` que fornece métodos estáticos para gerar diversos tipos de gráficos a partir de um `pandas.DataFrame`. Cada método recebe:
- `data`: um `DataFrame` cujas colunas são usadas como dados. A primeira coluna é o Y e as restantes são o X.
- `kwargs` (opcional): parâmetros adicionais do Matplotlib.

Métodos disponíveis:
- `line_chart`: gráfico de linhas (várias séries X vs Y).
- `scatter_chart`: diagrama de dispersão (várias séries X vs Y).
- `bar_chart`: gráfico de barras (valores da primeira coluna por categoria).
- `histogram`: histograma da primeira coluna.
- `box_plot`: diagrama de caixa para todas as colunas.
- `pie_chart`: gráfico de pizza da primeira coluna.
- `heatmap`: mapa de calor da matriz de correlação.
- `area_chart`: gráfico de área empilhada para colunas selecionadas.
- `scatter_matrix`: matriz de dispersão (pair plot) de todas as colunas.

Uso básico:

```python
from ChartGenerator import ChartGenerator
import pandas as pd

df = pd.read_csv("dados.csv")
ChartGenerator(df).line_chart("colX", "colY", color="blue", linewidth=2)
```


