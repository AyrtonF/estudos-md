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