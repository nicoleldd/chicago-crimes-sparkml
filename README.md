# Chicago Crimes: prevendo se uma ocorrência termina em prisão (Spark ML)

Trabalho de classificação binária com **PySpark** e **Spark ML** sobre as ocorrências criminais de Chicago (2020–2025). A pergunta é:

> Dadas as informações de uma ocorrência (natureza do crime, local, território, data e hora), é possível prever se ela termina em **prisão** (`arrest`)?

O projeto está em **um único notebook**, `Trabalho_SparkML_ChicagoCrimes.ipynb`, pensado para rodar no **Google Colab**.

- **Autor(a):** *(preencher nome completo)*
- **Disciplina / Instituição:** *(preencher)*
- **Link do notebook no Colab:** *(colar o link)*

---

## Dados

- **Fonte:** [Crimes – 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2), portal de dados abertos da Cidade de Chicago (API SODA, dataset `ijzp-q8t2`).
- **Recorte:** ocorrências de 2020 a 2025 (2026 fica de fora por ser um ano incompleto).
- **Alvo:** coluna `arrest` (houve prisão ou não). A classe positiva é **minoritária**, então o problema é desbalanceado.
- **Os dados não estão neste repositório.** O notebook os baixa automaticamente da API (um CSV por ano) e confere a contagem de linhas lida pelo Spark contra a contagem informada pela API. A base é atualizada diariamente, então os números podem mudar levemente entre execuções (o notebook imprime os valores da execução).

## Tecnologias

| Uso | Ferramenta |
|---|---|
| Processamento distribuído | **PySpark** (`SparkSession`, DataFrames, SQL functions) |
| Modelagem e avaliação | **Spark ML** (`pyspark.ml`): `Pipeline`, `LogisticRegression`, `RandomForestClassifier`, `CrossValidator`, `TrainValidationSplit`, `BinaryClassificationEvaluator`, `MulticlassClassificationEvaluator` |
| Gráficos e tabelas pequenas | pandas, NumPy, matplotlib (recebem só resultados já agregados pelo Spark) |
| Ambiente | Google Colab (Spark em modo local, `local[*]`) |

## O que o notebook faz

1. **Contextualização:** objetivo, dicionário de dados e roteiro.
2. **Configuração e ingestão:** sessão Spark, download pela API SODA, leitura com **schema manual** e conferência da leitura contra a fonte.
3. **Qualidade e exploração:** estatísticas descritivas, valores ausentes, testes de consistência, duplicatas, desbalanceamento do alvo e gráficos exploratórios.
4. **Paralelismo:** partições, *data skew* e experimento medindo o tempo de um *shuffle* com diferentes valores de `spark.sql.shuffle.partitions`.
5. **Engenharia de atributos e `Pipeline`:** divisão treino/teste, **pesos de classe** para o desbalanceamento e um único `Pipeline` com imputação, *one-hot*, atributos cíclicos de hora/dia/mês e padronização.
6. **Modelagem:** ajuste de hiperparâmetros por validação (cruzada na regressão logística e `TrainValidationSplit` na Random Forest) e *baselines*.
7. **Avaliação:** AUC-ROC, AUC-PR, precisão, recall, F1, matrizes de confusão, curvas ROC e Precisão×Recall (calculadas no Spark), sensibilidade ao limiar, importância de variáveis e coeficientes, desempenho por tipo de ocorrência, **validação temporal** (treina no passado, testa no último ano) e **ablação** (quanto cada grupo de atributos acrescenta).
8. **Persistência:** `PipelineModel` salvo, recarregado e conferido (predições idênticas), mais um exemplo de uso com uma ocorrência nova.
9. **Conclusão, limitações e melhorias futuras.**

## Decisões metodológicas

- **Sem vazamento de dados:** todo o pré-processamento e os pesos de classe são aprendidos **só com o treino**; no `CrossValidator` o `Pipeline` inteiro é refeito em cada *fold*.
- **Métrica principal: AUC-PR**, mais exigente que a acurácia (que engana com classes desbalanceadas) e mais adequada ao desbalanceamento.
- **Avaliação na distribuição real:** os pesos de classe são usados no treino, nunca na avaliação.
- **Comparação com *baselines*:** (1) sempre prever "sem prisão" e (2) regressão logística só com o tipo de crime.
- **Honestidade sobre limitações:** o código do crime (`iucr`, `fbi_code`) é parcialmente definido junto com a ação policial, então os resultados descrevem **como a polícia atua e registra**, e não uma relação causal. As probabilidades dos modelos com pesos **não são calibradas**.
- **Interpretação gerada pelo código:** os números dos textos de análise são impressos por células de código ("Leitura automática"), então o texto sempre bate com os resultados da execução.

## Como executar

1. Abra o notebook no **Google Colab** (ou envie o `.ipynb` para o Colab).
2. Use um ambiente com **Python 3** padrão do Colab. O PySpark é instalado automaticamente se estiver faltando. Não é preciso dado local nem credencial.
3. Menu **Ambiente de execução → Executar tudo**.

**Parâmetros** (célula "Parâmetros do trabalho", seção 2.1):

| Parâmetro | Padrão | Para que serve |
|---|---|---|
| `FRACAO_DADOS` | `1.0` | Use `0.1` para testar o notebook rapidamente. **Deixe `1.0` na versão final.** |
| `EXECUTAR_TUNING` | `True` | Liga/desliga a busca de hiperparâmetros. |
| `FRACAO_TUNING` | `0.25` | Fração do treino usada na busca de hiperparâmetros. |
| `NUM_ARVORES_FINAL` | `60` | Árvores da Random Forest final. |

**Tempo:** a execução completa leva em torno de **uma hora** no Colab gratuito (2 núcleos), principalmente por causa da Random Forest. O download dos dados depende da conexão.

## Estrutura do repositório

```
.
├── README.md
└── Trabalho_SparkML_ChicagoCrimes.ipynb   # notebook completo (com as saídas da execução)
```

Os dados (`/content/chicago_crimes`) e os modelos treinados (`/content/modelos`) são gerados em tempo de execução no Colab e **não** são versionados.

## Limitações

- Spark em **modo local** com poucos núcleos: a escalabilidade para um cluster é demonstrada pelo desenho do código, não por medições em cluster.
- A busca de hiperparâmetros usa uma fração do treino e grades pequenas, para caber no Colab; o notebook avisa quando um valor ótimo fica na borda da grade.
- A escolha entre regressão logística e Random Forest usa o conjunto de teste (diferença pequena); o ideal seria decidir em um conjunto de validação separado.
- Os resultados refletem registros e práticas policiais de Chicago e não devem ser usados para decisões sobre pessoas.

## Referências

- Cidade de Chicago, *Crimes – 2001 to Present* (dados abertos): https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
- Documentação do Apache Spark MLlib: https://spark.apache.org/docs/latest/ml-guide.html
