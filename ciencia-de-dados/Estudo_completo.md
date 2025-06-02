# Estudo Didático: Teorema de Bayes e Classificador Naive Bayes

---

## 1. O que é o Teorema de Bayes?

O Teorema de Bayes é uma fórmula fundamental em probabilidade e estatística que nos ajuda a atualizar nossas **crenças** (probabilidades) quando recebemos uma nova informação ou evidência.

### Fórmula do Teorema de Bayes

\[
P(A|B) = \frac{P(A) \times P(B|A)}{P(B)}
\]

- **P(A)**: Probabilidade de A ocorrer (antes de saber B) — **probabilidade a priori**.
- **P(B|A)**: Probabilidade de B ocorrer, dado que A ocorreu — **verossimilhança**.
- **P(B)**: Probabilidade de B ocorrer (em geral).
- **P(A|B)**: Probabilidade de A ocorrer, dado que B foi observado — **probabilidade a posteriori**.

---

### Exemplo Prático com Moedas

Imagine uma caixa com duas moedas:

- **Moeda Justa**: 50% de chance de dar "Cara" (P(Cara|Justa) = 0,5)
- **Moeda Viciada**: 100% de chance de dar "Cara" (P(Cara|Viciada) = 1)

Escolhemos **aleatoriamente** uma moeda (50% para cada) e jogamos uma vez.

#### Passo 1: Qual a chance de dar "Cara" no geral?

\[
P(Cara) = P(Moeda\ Justa) \times P(Cara|Justa) + P(Moeda\ Viciada) \times P(Cara|Viciada)
\]
\[
P(Cara) = 0,5 \times 0,5 + 0,5 \times 1 = 0,25 + 0,5 = 0,75
\]

**Ou seja, 75% de chance de dar "Cara".**

#### Passo 2: Se soubermos que a moeda é justa, qual a chance de dar "Cara"?

\[
P(Cara|Moeda\ Justa) = 0,5 = 50\%
\]

#### Passo 3: Se der "Cara", qual a chance da moeda ser a viciada?

Aqui usamos o Teorema de Bayes:

\[
P(Viciada|Cara) = \frac{P(Viciada) \times P(Cara|Viciada)}{P(Cara)}
\]
\[
P(Viciada|Cara) = \frac{0,5 \times 1}{0,75} = \frac{0,5}{0,75} \approx 0,666\ (\text{66,6%})
\]

---

## 2. Classificador Naive Bayes

É um método de Machine Learning que usa o Teorema de Bayes para **classificar dados** (por exemplo, decidir se um e-mail é spam ou não).

### Ideia Central

- "Naive" (Ingênuo): Supõe que todas as características (features) são **independentes** — mesmo que nem sempre isso seja verdade.
- Classifica escolhendo a classe de maior probabilidade, considerando as features do exemplo.

### Fórmula do Naive Bayes

\[
P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}
\]

- **C**: Classe (exemplo: "spam", "não spam", "doente", "saudável").
- **X**: Vetor de características (features).

#### Como funciona

1. Calcula a probabilidade de cada classe com base nas features do exemplo.
2. Assume independência entre as features:
   \[
   P(X|C) = P(x_1|C) \times P(x_2|C) \times ... \times P(x_n|C)
   \]
3. Escolhe a classe com **maior probabilidade**.

---

### Tipos de Naive Bayes

- **GaussianNB**: Para dados contínuos (ex: altura, pressão arterial). Assume distribuição **normal**.
- **MultinomialNB**: Para contagem de eventos (ex: frequência de palavras). Muito usado em textos.
- **BernoulliNB**: Para dados binários (ex: presença/ausência de uma palavra).

---

#### Exemplo Numérico: Diagnóstico de Saúde

Temos pacientes com duas classes: "doente" e "saudável".

- **Doentes**: Pressão média = 150, variância = 25
- **Saudáveis**: Pressão média = 120, variância = 16

Novo paciente com pressão = 140.

Para cada classe, usamos a fórmula da **distribuição normal**:

\[
P(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\]

- Para "doente": \(\mu=150\), \(\sigma^2=25\), \(x=140\)
- Para "saudável": \(\mu=120\), \(\sigma^2=16\), \(x=140\)

Basta calcular os dois valores e comparar!

---

## 3. Quando Usar Naive Bayes?

### Vantagens

- **Rápido** e eficiente, mesmo com muitos dados.
- Funciona bem com pouco treinamento.
- Simples de implementar.

### Desvantagens

- A suposição de independência pode não valer (features correlacionadas prejudicam o modelo).
- Pode perder para outros métodos (ex: Regressão Logística) se as features não forem independentes.

### Usos Comuns

- Filtro de spam.
- Diagnóstico médico.
- Classificação de textos.
- Sistemas de recomendação simples.

---

## 4. Comparação com Outros Métodos

### Regressão Logística

- Calcula **pesos** para cada feature (não assume independência).
- Melhor quando as features estão correlacionadas.
- Mais flexível.

### Naive Bayes

- Mais **simples** e **rápido**.
- Pode ser melhor com poucos dados ou features independentes.

---

### Exemplo de Código Python

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Treinando GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Treinando Regressão Logística
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Comparando acurácias
print("Acurácia GaussianNB:", gnb.score(X_test, y_test))
print("Acurácia Regressão Logística:", lr.score(X_test, y_test))
```
**Resultado:** Depende dos dados! Um pode superar o outro em diferentes situações.

---

## 5. Resumo Final

- **Teorema de Bayes:** Atualiza probabilidades com novas evidências.
- **Naive Bayes:** Classificador simples que usa Bayes + suposição de independência.
- **GaussianNB:** Dados contínuos.
- **MultinomialNB:** Contagens (ex: textos).
- **Quando usar?** Para modelos rápidos e simples, especialmente se as features forem quase independentes.

---

## Dicas para Prova

- Sempre identifique se as features são dependentes ou independentes!
- Em problemas práticos, escreva os passos do cálculo.
- Lembre: Naive Bayes é ótimo quando você quer **rapidez** e **simplicidade**.
- Se pedir código, cite a importação de `GaussianNB`, `MultinomialNB` ou `LogisticRegression` do `sklearn`.

---


# Estudo Didático: Mineração de Dados, Modelos Supervisionados e Não Supervisionados

---

## 1. O que é Mineração de Dados?

Mineração de Dados é o processo de explorar grandes volumes de dados para descobrir padrões, tendências e relações que não são óbvias à primeira vista. É como ser um detetive de dados!

**Exemplo prático:**  
Em supermercados, se muitos clientes compram "pão + manteiga" juntos, o gerente pode posicionar esses itens próximos para aumentar as vendas.

---

## 2. Passos Básicos da Mineração de Dados (Pipeline)

1. **Coleta:** Obtenção dos dados (planilhas, bancos de dados, logs, etc.).
2. **Limpeza:** Tratamento de dados ausentes, erros ou duplicidades (ex: preencher idade média onde faltar).
3. **Mineração:** Aplicação de algoritmos para extrair padrões.
4. **Avaliação:** Verificação se os padrões encontrados fazem sentido.
5. **Uso:** Aplicação dos padrões no mundo real (ex: recomendações automáticas).

---

## 3. Técnicas Principais

### 3.1 Classificação (“Separar em Categorias”)

- **Objetivo:** Prever a qual classe um dado pertence (ex: e-mail “spam” ou “não spam”).
- **Algoritmo clássico:** Regressão Logística.

#### Como funciona a Regressão Logística?

- Transforma a combinação linear das variáveis em uma probabilidade entre 0 e 1 usando a Função Sigmoide.

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

- Se \(\sigma(z) \geq 0.5\): classifica como “1” (ex: sobreviveu).
- Se \(\sigma(z) < 0.5\): classifica como “0” (ex: não sobreviveu).

**Exemplo (Titanic):**

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.predict([[1, 0, 25, 50]]))  # prevendo para um homem, 25 anos, 1ª classe, passagem 50
```

---

### 3.2 Regressão (“Prever Números”)

- **Objetivo:** Estimar valores contínuos (ex: preço de uma casa).
- **Algoritmo clássico:** Regressão Linear.

\[
\hat{y} = m x + c
\]

- \(m\): inclinação da reta.
- \(c\): intercepto (onde corta o eixo y).

**Exemplo:**

Se \(m = 3\), \(c = 4\), para \(x = 2\):

\[
\hat{y} = 3 \times 2 + 4 = 10
\]

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print(model.predict([[10]]))  # prevê preço para casa de 10m²
```

---

### 3.3 Clustering (“Agrupar sem Rótulos”)

- **Objetivo:** Encontrar grupos naturais nos dados (ex: perfis de clientes).
- **Algoritmo clássico:** k-Means.

**Passos:**
1. Escolhe \(k\) (ex: 3 grupos).
2. Inicializa centroides.
3. Cada ponto é atribuído ao centroide mais próximo.
4. Atualiza centroides (média dos pontos do cluster).
5. Repete até estabilizar.

**Exemplo:**

Dados: \(X = [1, 2, 8, 10]\), \(k = 2\).

- Inicial: Centroides em 1 e 2.
- Iteração: [1] perto de 1, [2,8,10] perto de 2.
- Novo centroide: média de [2,8,10] = (2+8+10)/3 = 6.67.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
```

---

### 3.4 Regras de Associação (“Descobrir Combinações”)

- **Objetivo:** Encontrar itens que aparecem juntos (ex: leite + pão).

**Métricas:**
- **Suporte:** Freq. que um item aparece.
\[
Support(A) = \frac{\text{transações com A}}{\text{total de transações}}
\]
- **Confiança:** Probabilidade de B dado A.
\[
Confidence(A \rightarrow B) = \frac{Support(A \cup B)}{Support(A)}
\]
- **Lift:** Relação entre A e B.
\[
Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)}
\]
- Se Lift > 1: A e B relacionados.

**Exemplo:**
- 3 transações com leite em 10 → Suporte = 0.3
- 2 transações com leite + pão → Suporte(leite ∪ pão) = 0.2
- Confiança(leite → pão) = 0.2/0.3 ≈ 0.67
- Suporte(pão) = 0.4 → Lift = 0.67/0.4 = 1.675

```python
from mlxtend.frequent_patterns import apriori
rules = apriori(transactions, min_support=0.5)
```

---

## 4. Pré-Processamento de Dados

Antes de minerar, é preciso arrumar os dados!

### 4.1 Dados Faltantes

- Substituir pela média/mediana:
```python
df['idade'].fillna(df['idade'].mean(), inplace=True)
```

### 4.2 Escalonamento (Standardização)

- Padroniza para média 0 e desvio padrão 1:

\[
x' = \frac{x - \bar{x}}{s}
\]

**Exemplo:**  
Dados = [10, 20, 30], Média = 20, Desvio padrão ≈ 8.16.  
Para x=30:

\[
x' = \frac{30-20}{8.16} \approx 1.22
\]

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

### 4.3 One-Hot Encoding

- Transforma categorias em colunas binárias.
| cor      | cor_vermelho | cor_azul | cor_verde |
|----------|--------------|----------|-----------|
| vermelho | 1            | 0        | 0         |
| azul     | 0            | 1        | 0         |

```python
pd.get_dummies(df, columns=['cor'])
```

---

## 5. Quando Usar Cada Técnica?

- **Classificação:** Quando as categorias são conhecidas (spam/não spam, doença/não doença).
- **Regressão:** Para prever valores contínuos (preços, temperaturas).
- **Clustering:** Para encontrar grupos ocultos (segmentação de clientes).
- **Regras de Associação:** Para identificar combinações comuns (mercado, recomendações).

---

## 6. Resumo Visual (Imagem Mental)

Imagine os dados como uma mina de ouro:
- **Classificação:** Separa o ouro em “puro” e “impuro”.
- **Regressão:** Pesa o ouro para saber seu valor.
- **Clustering:** Agrupa pedras semelhantes.
- **Regras de Associação:** Descobre que “ouro + prata” aparecem juntos.

---

# PARTE 2 — Destrinchando Fórmulas e Cálculos

---

## 1. Regressão Logística (Classificação)

### 1.1 Função Sigmoide

Converte qualquer número real \(z\) em uma probabilidade entre 0 e 1:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**Exemplo:**

Se \(z = 2\):

\[
e^{-2} \approx 0.135
\]
\[
\sigma(2) = \frac{1}{1 + 0.135} \approx \frac{1}{1.135} \approx 0.88
\]

### 1.2 Cálculo do \(z\)

\[
z = \beta_1x_1 + \beta_2x_2 + \dots + \beta_dx_d + b
\]

- \(\beta_i\): Peso aprendido para cada variável \(x_i\).
- \(b\): Viés (intercepto).

**Exemplo prático (Titanic):**

- \(\beta_{classe} = -1.2\)
- \(\beta_{sexo} = 2.5\)
- \(\beta_{idade} = -0.1\)
- \(b = 0.3\)

Passageiro: 1ª classe (\(x_1=1\)), mulher (\(x_2=1\)), idade 30 (\(x_3=30\)):

\[
z = (-1.2 \times 1) + (2.5 \times 1) + (-0.1 \times 30) + 0.3 = -1.2 + 2.5 - 3 + 0.3 = -1.4
\]

\[
\sigma(-1.4) \approx 0.20
\]
Interpretação: 20% de chance de sobreviver.

---

## 2. Regressão Linear

### 2.1 Fórmula da reta

\[
\hat{y} = m x + c
\]

**Exemplo:**  
\(m = 3\), \(c = 4\), \(x = 2\):

\[
\hat{y} = 3 \times 2 + 4 = 10
\]

### 2.2 Erro Quadrático Médio (MSE)

Avalia a qualidade do modelo:

\[
J = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
\]

**Exemplo:**  
Previsões: [10, 12], reais: [9, 14]

\[
J = \frac{(10-9)^2 + (12-14)^2}{2} = \frac{1 + 4}{2} = 2.5
\]

---

## 3. k-Means (Clustering)

### 3.1 Passos do Algoritmo

1. Inicialização: Escolhe \(k\) centroides aleatórios.
2. Atribuição: Cada ponto vai para o centroide mais próximo.
3. Atualização: Recalcula centroides como média dos pontos do cluster.
4. Repete até os centroides não mudarem.

**Exemplo Numérico:**  
Dados: \(X = [1, 2, 8, 10]\), \(k = 2\).

- **Passo 1:** Centroides iniciais: \(C_1=1\), \(C_2=2\)
- **Passo 2:** Atribuição:
  - Próximo de 1: [1]
  - Próximo de 2: [2,8,10]
- **Passo 3:** Novos centroides:
  - \(C_1\): média de [1] = 1
  - \(C_2\): média de [2,8,10] = (2+8+10)/3 = 6.67
- Repete até convergir.

---

## 4. Regras de Associação (Apriori)

### 4.1 Métricas

- **Suporte:**  
\[
Support(A) = \frac{\text{Nº de transações com A}}{\text{Nº total de transações}}
\]
**Exemplo:**  
3 transações com leite em 10  
\(Support(leite) = \frac{3}{10} = 0.3\)

- **Confiança:**
\[
Confidence(A \rightarrow B) = \frac{Support(A \cup B)}{Support(A)}
\]
**Exemplo:**  
\(Support(leite \cup pão) = 0.2\), \(Support(leite) = 0.3\)  
\(Confidence(leite \rightarrow pão) = 0.2 / 0.3 \approx 0.67\)

- **Lift:**
\[
Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)}
\]
**Exemplo:**  
\(Support(pão) = 0.4\)  
\(Lift(leite \rightarrow pão) = 0.67 / 0.4 = 1.675\) (relação positiva)

---

## 5. Pré-Processamento

### 5.1 Escalonamento (StandardScaler)

\[
x' = \frac{x - \bar{x}}{s}
\]

**Exemplo:**  
Dados: [10, 20, 30], Média = 20, s ≈ 8.16  
Para x=30:  
\(x' = (30-20)/8.16 \approx 1.22\)

### 5.2 One-Hot Encoding

| cor      | cor_vermelho | cor_azul |
|----------|--------------|----------|
| vermelho | 1            | 0        |
| azul     | 0            | 1        |

---

## 6. Resumo Visual das Técnicas

| Técnica                 | Fórmula Chave            | Exemplo de Uso                   |
|-------------------------|--------------------------|----------------------------------|
| Regressão Logística     | \(\sigma(z) = \frac{1}{1 + e^{-z}}\) | Diagnóstico médico (0 ou 1)      |
| Regressão Linear        | \(\hat{y} = mx + c\)     | Prever preços de imóveis         |
| k-Means                 | Atualização de centroides | Segmentação de clientes          |
| Apriori (Regras)        | \(Lift = \frac{Confidence}{Support}\) | Análise de mercado (cesta)       |

---

**DICA PARA PROVA:**  
- Sempre escreva as fórmulas, explique cada termo e faça um exemplo numérico!
- Entenda o objetivo de cada técnica e o tipo de dado em que ela melhor se aplica.

---

# Estudo Didático: Testes de Hipóteses Estatísticas

---

## 1. Introdução aos Testes de Hipóteses

Testes de hipóteses são ferramentas estatísticas que nos ajudam a tomar decisões baseadas em dados.  
Você propõe uma hipótese (chamada de hipótese nula, H₀) e, com base nos dados, decide se há evidência suficiente para rejeitá-la em favor da hipótese alternativa (H₁).

### Exemplo prático
Um hospital quer saber se um novo remédio reduz o tempo de recuperação.

- **Hipótese Nula (H₀):** O remédio não tem efeito (tempo médio = 20 dias)
- **Hipótese Alternativa (H₁):** O remédio tem efeito (tempo médio ≠ 20 dias)

---

## 2. Teste t para Uma Amostra

### Quando usar?
Quando você quer comparar a média de uma amostra com um valor conhecido/histórico.

### Fórmula da Estatística t

\[
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
\]

- \(\bar{x}\) = Média da amostra
- \(\mu_0\) = Média esperada sob H₀
- \(s\) = Desvio padrão da amostra
- \(n\) = Tamanho da amostra

#### Passo a Passo

1. Calcule a média (\(\bar{x}\)) e o desvio padrão (\(s\)) dos dados.
2. Subtraia a média esperada (\(\mu_0\)) da média da amostra.
3. Divida o desvio padrão pelo raiz quadrada do tamanho da amostra (\(s / \sqrt{n}\)).
4. Divida o resultado do passo 2 pelo do passo 3.

#### Exemplo Numérico

- **Amostra:** 30 pacientes  
- \(\bar{x}\) = 17 dias  
- \(s\) = 2.5  
- \(n\) = 30  
- **H₀:** \(\mu_0 = 20\) dias

\[
t = \frac{17 - 20}{2.5 / \sqrt{30}} = \frac{-3}{2.5 / 5.477} = \frac{-3}{0.456} \approx -6.58
\]

**Interpretação:**  
O valor t indica que a média observada está 6.58 desvios padrões abaixo da média nula.

**Valor-p:**  
É a probabilidade de obter um resultado tão extremo quanto o observado, assumindo que H₀ é verdadeira.  
Se \(p < 0.05\): rejeitamos H₀.

#### Código Python

```python
from scipy import stats
dados = [16, 18, 17, 15, ...]  # Valores da amostra
t_stat, p_valor = stats.ttest_1samp(dados, popmean=20)
print(f"t = {t_stat:.2f}, p = {p_valor:.4f}")
```

**Conclusão:**  
Se \(p < 0.05\), rejeitamos H₀. Ou seja, o remédio reduz o tempo de recuperação.

---

## 3. Teste t para Duas Amostras Independentes

### Quando usar?
Quando você quer comparar as médias de dois grupos distintos (ex: controle vs. tratado).

### Fórmula

\[
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
\]

- \(\bar{x}_1, \bar{x}_2\): médias dos grupos 1 e 2
- \(s_1, s_2\): desvios padrões dos grupos
- \(n_1, n_2\): tamanhos dos grupos

#### Passo a Passo

1. Calcule a média e o desvio padrão de cada grupo.
2. Subtraia as médias (\(\bar{x}_1 - \bar{x}_2\)).
3. Calcule \(\frac{s_1^2}{n_1}\) e \(\frac{s_2^2}{n_2}\), some e tire a raiz quadrada.
4. Divida a diferença de médias pelo resultado do passo 3.

#### Exemplo

- Grupo A: \(\bar{x}_1 = 20\), \(s_1 = 2.5\), \(n_1 = 30\)
- Grupo B: \(\bar{x}_2 = 18\), \(s_2 = 2.3\), \(n_2 = 30\)

\[
t = \frac{20 - 18}{\sqrt{\frac{2.5^2}{30} + \frac{2.3^2}{30}}}
= \frac{2}{\sqrt{\frac{6.25}{30} + \frac{5.29}{30}}}
= \frac{2}{\sqrt{0.2083 + 0.1763}}
= \frac{2}{\sqrt{0.3846}}
= \frac{2}{0.620}
\approx 3.23
\]

**Interpretação:**  
Se \(t\) é alto (positivo ou negativo), indica diferença nas médias.

**Valor-p:**  
Se \(p < 0.05\): diferença estatisticamente significativa.

#### Código Python

```python
from scipy import stats
grupo_A = [19, 21, 20, ...]
grupo_B = [17, 18, 19, ...]
t_stat, p_valor = stats.ttest_ind(grupo_A, grupo_B)
print(f"t = {t_stat:.2f}, p = {p_valor:.4f}")
```

**Conclusão:**  
Se \(p < 0.05\), o novo tratamento é mais eficaz.

---

## 4. Teste Qui-Quadrado de Independência

### Quando usar?
Para verificar se existe associação entre duas variáveis categóricas (ex: plano de saúde vs. comparecimento).

### Fórmula

\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]

- \(O\): valores observados
- \(E\): valores esperados sob independência

#### Passo a Passo

1. Monte a tabela de contingência (linhas = categoria 1, colunas = categoria 2).
2. Calcule o esperado para cada célula:
   - \(E = \frac{\text{Total linha} \times \text{Total coluna}}{\text{Total geral}}\)
3. Para cada célula, calcule \((O - E)^2 / E\).
4. Some os valores para obter \(\chi^2\).

#### Exemplo

|           | Compareceu | Não Compareceu | Total |
|-----------|------------|----------------|-------|
| Plano A   | 40         | 10             | 50    |
| Plano B   | 25         | 15             | 40    |
| Total     | 65         | 25             | 90    |

**Cálculo para Plano A, Compareceu:**
\[
E = \frac{50 \times 65}{90} \approx 36.11
\]

**Cálculo para Plano A, Não Compareceu:**
\[
E = \frac{50 \times 25}{90} \approx 13.89
\]

**Agora, para cada célula:**
\[
\chi^2 = \sum \frac{(O - E)^2}{E}
\]
\[
\chi^2 = \frac{(40-36.11)^2}{36.11} + \frac{(10-13.89)^2}{13.89} + ... \approx 5.12
\]

**Valor-p:**  
Se \(p < 0.05\), as variáveis são associadas.

#### Código Python

```python
import pandas as pd
from scipy.stats import chi2_contingency

tabela = pd.DataFrame({
    'Compareceu': [40, 25],
    'Não Compareceu': [10, 15]
})
chi2, p, dof, esperado = chi2_contingency(tabela)
print(f"χ² = {chi2:.2f}, p = {p:.4f}")
```

**Conclusão:**  
Se \(p < 0.05\), o tipo de plano afeta o comparecimento.

---

## 5. Teste de Proporções

### Quando usar?
Para comparar uma proporção observada com uma esperada.

### Fórmula

\[
z = \frac{\hat{p} - p_0}{\sqrt{ \frac{p_0(1-p_0)}{n} } }
\]

- \(\hat{p}\): proporção observada (ex: 85/100 = 0.85)
- \(p_0\): proporção esperada (ex: 0.70)
- \(n\): tamanho da amostra

#### Passo a Passo

1. Calcule a proporção observada (\(\hat{p}\)).
2. Subtraia a proporção esperada (\(p_0\)).
3. Calcule o denominador: \(p_0(1-p_0)/n\), tire a raiz quadrada.
4. Divida o resultado do passo 2 pelo do passo 3.

#### Exemplo

- Sucesso: 85 de 100 (\(\hat{p} = 0.85\))
- Esperado: \(p_0 = 0.70\)
- \(n = 100\)

\[
z = \frac{0.85 - 0.70}{ \sqrt{ \frac{0.7 \times 0.3}{100} } }
= \frac{0.15}{ \sqrt{ 0.0021 } }
= \frac{0.15}{0.0458}
\approx 3.27
\]

**Valor-p:**  
Se \(p < 0.05\), a proporção observada é significativamente maior que a esperada.

#### Código Python

```python
from statsmodels.stats.proportion import proportions_ztest

stat, pval = proportions_ztest(count=85, nobs=100, value=0.7, alternative='larger')
print(f"z = {stat:.2f}, p = {pval:.4f}")
```

**Conclusão:**  
Se \(p < 0.05\), a proporção de sucesso é maior que a esperada.

---

## 6. Resumo dos Testes

| Teste                 | Quando Usar?                | Fórmula Chave                | Decisão (p < 0.05)           |
|-----------------------|-----------------------------|------------------------------|------------------------------|
| t uma amostra         | Média amostral vs. valor    | \( t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \) | Rejeita H₀                   |
| t duas amostras       | Médias de 2 grupos          | \( t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} \) | Diferença significativa       |
| Qui-quadrado          | Assoc. entre categorias     | \( \chi^2 = \sum \frac{(O-E)^2}{E} \) | Variáveis associadas          |
| Proporções            | Proporção observada vs. esp.| \( z = \frac{\hat{p} - p_0}{\sqrt{ \frac{p_0(1-p_0)}{n} } } \) | Proporção diferente do esperado |

---

## 7. Dicas Finais

- Sempre escreva as hipóteses (H₀ e H₁) antes de calcular!
- Mostre cada passo do cálculo.
- Valor-p < 0.05 = resultado significativo (rejeite H₀!).
- Cite que os testes partem do pressuposto de amostras aleatórias e independentes.
- Use exemplos numéricos para mostrar seu raciocínio na prova.