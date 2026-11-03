# Chicago Crimes: prevendo se uma ocorrência termina em prisão (Spark ML)

Trabalho de classificação binária com **PySpark** e **Spark ML** sobre as ocorrências criminais de Chicago (2020–2025). A pergunta é:

> Dadas as informações de uma ocorrência (natureza do crime, local, território, data e hora), é possível prever se ela termina em **prisão** (`arrest`)?

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

