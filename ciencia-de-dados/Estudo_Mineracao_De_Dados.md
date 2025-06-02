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

```
sigma(z) = 1 / (1 + e^(-z))
```

- Se sigma(z) >= 0.5: classifica como “1” (ex: sobreviveu).
- Se sigma(z) < 0.5: classifica como “0” (ex: não sobreviveu).

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

```
y_hat = m * x + c
```

- m: inclinação da reta.
- c: intercepto (onde corta o eixo y).

**Exemplo:**

Se m = 3, c = 4, para x = 2:

```
y_hat = 3 * 2 + 4 = 10
```

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
1. Escolhe k (ex: 3 grupos).
2. Inicializa centroides.
3. Cada ponto é atribuído ao centroide mais próximo.
4. Atualiza centroides (média dos pontos do cluster).
5. Repete até estabilizar.

**Exemplo:**

Dados: X = [1, 2, 8, 10], k = 2.

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

```
Support(A) = (transações com A) / (total de transações)
```
- **Confiança:** Probabilidade de B dado A.

```
Confidence(A -> B) = Support(A ∪ B) / Support(A)
```
- **Lift:** Relação entre A e B.

```
Lift(A -> B) = Confidence(A -> B) / Support(B)
```
- Se Lift > 1: A e B relacionados.

**Exemplo:**
- 3 transações com leite em 10 → Suporte = 0.3
- 2 transações com leite + pão → Suporte(leite ∪ pão) = 0.2
- Confiança(leite -> pão) = 0.2/0.3 ≈ 0.67
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

```
x' = (x - x̄) / s
```

**Exemplo:**  
Dados = [10, 20, 30], Média = 20, Desvio padrão ≈ 8.16.  
Para x=30:

```
x' = (30 - 20) / 8.16 ≈ 1.22
```

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

Converte qualquer número real z em uma probabilidade entre 0 e 1:

```
sigma(z) = 1 / (1 + e^(-z))
```

**Exemplo:**

Se z = 2:

```
e^(-2) ≈ 0.135
sigma(2) = 1 / (1 + 0.135) ≈ 1 / 1.135 ≈ 0.88
```

### 1.2 Cálculo do z

```
z = beta1*x1 + beta2*x2 + ... + betad*xd + b
```

- betai: Peso aprendido para cada variável xi.
- b: Viés (intercepto).

**Exemplo prático (Titanic):**

- beta_classe = -1.2
- beta_sexo = 2.5
- beta_idade = -0.1
- b = 0.3

Passageiro: 1ª classe (x1=1), mulher (x2=1), idade 30 (x3=30):

```
z = (-1.2 * 1) + (2.5 * 1) + (-0.1 * 30) + 0.3
z = -1.2 + 2.5 - 3 + 0.3 = -1.4
```

```
sigma(-1.4) ≈ 0.20
```
Interpretação: 20% de chance de sobreviver.

---

## 2. Regressão Linear

### 2.1 Fórmula da reta

```
y_hat = m * x + c
```

**Exemplo:**  
m = 3, c = 4, x = 2:

```
y_hat = 3 * 2 + 4 = 10
```

### 2.2 Erro Quadrático Médio (MSE)

Avalia a qualidade do modelo:

```
J = (1/N) * sum((y_hat_i - y_i)^2)
```

**Exemplo:**  
Previsões: [10, 12], reais: [9, 14]

```
J = ((10-9)^2 + (12-14)^2) / 2 = (1 + 4) / 2 = 2.5
```

---

## 3. k-Means (Clustering)

### 3.1 Passos do Algoritmo

1. Inicialização: Escolhe k centroides aleatórios.
2. Atribuição: Cada ponto vai para o centroide mais próximo.
3. Atualização: Recalcula centroides como média dos pontos do cluster.
4. Repete até os centroides não mudarem.

**Exemplo Numérico:**  
Dados: X = [1, 2, 8, 10], k = 2.

- Passo 1: Centroides iniciais: C1 = 1, C2 = 2
- Passo 2: Atribuição:
  - Próximo de 1: [1]
  - Próximo de 2: [2,8,10]
- Passo 3: Novos centroides:
  - C1: média de [1] = 1
  - C2: média de [2,8,10] = (2+8+10)/3 = 6.67
- Repete até convergir.

---

## 4. Regras de Associação (Apriori)

### 4.1 Métricas

- **Suporte:**  

```
Support(A) = (Nº de transações com A) / (Nº total de transações)
```
**Exemplo:**  
3 transações com leite em 10  
Support(leite) = 3/10 = 0.3

- **Confiança:**

```
Confidence(A -> B) = Support(A ∪ B) / Support(A)
```
**Exemplo:**  
Support(leite ∪ pão) = 0.2, Support(leite) = 0.3  
Confidence(leite -> pão) = 0.2 / 0.3 ≈ 0.67

- **Lift:**

```
Lift(A -> B) = Confidence(A -> B) / Support(B)
```
**Exemplo:**  
Support(pão) = 0.4  
Lift(leite -> pão) = 0.67 / 0.4 = 1.675 (relação positiva)

---

## 5. Pré-Processamento

### 5.1 Escalonamento (StandardScaler)

```
x' = (x - x̄) / s
```

**Exemplo:**  
Dados: [10, 20, 30], Média = 20, s ≈ 8.16  
Para x=30:  
x' = (30-20)/8.16 ≈ 1.22

### 5.2 One-Hot Encoding

| cor      | cor_vermelho | cor_azul |
|----------|--------------|----------|
| vermelho | 1            | 0        |
| azul     | 0            | 1        |

---

## 6. Resumo Visual das Técnicas

| Técnica                 | Fórmula Chave                  | Exemplo de Uso                   |
|-------------------------|-------------------------------|----------------------------------|
| Regressão Logística     | sigma(z) = 1 / (1 + e^(-z))   | Diagnóstico médico (0 ou 1)      |
| Regressão Linear        | y_hat = m * x + c             | Prever preços de imóveis         |
| k-Means                 | Atualização de centroides      | Segmentação de clientes          |
| Apriori (Regras)        | Lift = Confidence / Support    | Análise de mercado (cesta)       |

---

**DICA PARA PROVA:**  
- Sempre escreva as fórmulas, explique cada termo e faça um exemplo numérico!
- Entenda o objetivo de cada técnica e o tipo de dado em que ela melhor se aplica.
