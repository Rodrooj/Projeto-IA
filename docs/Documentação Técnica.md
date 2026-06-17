# Documentação Técnica — Tradutor de Libras para Áudio

## Arquitetura Atualizada com MediaPipe Holistic e Pipeline Híbrido CNN-1D + LSTM

---

# Resumo Executivo

Este documento apresenta a arquitetura atualizada do sistema **Tradutor de Libras para Áudio**, desenvolvida para corrigir limitações identificadas na versão inicial do projeto, incluindo:

* Baixa capacidade de generalização;
* Dificuldades de convergência durante o treinamento;
* Redução da acurácia em cenários reais;
* Ocorrência de *mode collapse* durante a inferência.

A nova solução combina:

* Extração de landmarks por meio do **MediaPipe Holistic**;
* Engenharia avançada de atributos com normalização espacial;
* Extração explícita de informações temporais;
* Classificação utilizando uma arquitetura híbrida **CNN-1D + LSTM**;
* Execução local (*Edge Computing*) via TensorFlow Lite.

O objetivo é proporcionar reconhecimento robusto de sinais em LIBRAS com baixa latência e alta precisão, preservando a privacidade dos usuários por meio do processamento local.

---

# 1. Visão Geral da Arquitetura

O sistema realiza a tradução em tempo real seguindo um pipeline composto por cinco etapas principais.

```text
Captura de Vídeo
        ↓
MediaPipe Holistic
        ↓
Engenharia de Atributos
(Normalização + Deltas)
        ↓
CNN-1D + LSTM
        ↓
Síntese de Voz
```

## 1.1 Fluxo Operacional

### 1. Captura de Vídeo

O sistema captura continuamente imagens da câmera do dispositivo em aproximadamente 30 FPS.

### 2. Extração de Landmarks

O MediaPipe Holistic identifica simultaneamente:

* Estrutura corporal superior;
* Mão esquerda;
* Mão direita.

Esses pontos são convertidos em coordenadas tridimensionais utilizadas pelo modelo de classificação.

### 3. Engenharia de Atributos

Antes da inferência, os dados passam por:

* Normalização espacial baseada na posição dos ombros;
* Cálculo dos deslocamentos temporais entre frames consecutivos.

### 4. Classificação

Uma janela temporal contendo 30 frames consecutivos é enviada para a rede neural híbrida CNN-1D + LSTM.

### 5. Síntese de Voz

Quando a confiança da predição ultrapassa 70%, a classe reconhecida é convertida em áudio por meio da Web Speech API.

---

# 2. Componentes Técnicos

## 2.1 Visão Computacional — MediaPipe Holistic

A adoção do MediaPipe Holistic resolve a principal limitação do protótipo original: a ausência de rastreamento detalhado dos dedos.

Como a configuração das mãos é um elemento fundamental da gramática da LIBRAS, a precisão da captura manual influencia diretamente a qualidade do reconhecimento.

### Estrutura de Landmarks

| Conjunto        | Quantidade | Dimensões | Finalidade                         |
| --------------- | ---------: | --------- | ---------------------------------- |
| Tronco Superior |  11 pontos | x, y, z   | Contexto postural e posicionamento |
| Mão Esquerda    |  21 pontos | x, y, z   | Configuração manual                |
| Mão Direita     |  21 pontos | x, y, z   | Configuração manual                |

### Total de Dados por Frame

```text
53 landmarks × 3 coordenadas = 159 atributos
```

Portanto, cada frame gera um vetor de entrada contendo **159 features**.

---

## 2.2 Engenharia de Atributos

A etapa de pré-processamento foi projetada para tornar o sistema mais robusto a variações de posicionamento e movimento.

### Normalização Espacial

O ponto médio entre os ombros é definido como a origem do sistema de coordenadas.

```text
Origem = (Ombro Esquerdo + Ombro Direito) / 2
```

Todas as demais coordenadas são recalculadas em relação a essa origem.

#### Benefícios

* Invariância à translação;
* Redução de ruído espacial;
* Maior capacidade de generalização.

---

### Deltas Temporais

Além das posições absolutas, o sistema calcula a diferença entre frames consecutivos:

```text
Δ = posição(t) − posição(t−1)
```

Essas informações fornecem ao modelo uma representação explícita do movimento.

#### Benefícios

* Captura da velocidade gestual;
* Captura da direção do movimento;
* Melhor distinção entre sinais semelhantes.

---

## 2.3 Arquitetura da Rede Neural

A arquitetura foi projetada para combinar extração espacial e análise temporal.

### Estrutura Geral

```text
Entrada (30,159)
        ↓
Conv1D
        ↓
LSTM 64
        ↓
LSTM 32
        ↓
Dense 128
        ↓
Softmax (10 Classes)
```

---

### Camada de Entrada

```text
Shape: (30, 159)
```

Representa:

* 30 frames consecutivos;
* 159 atributos por frame.

---

### Camada Conv1D

| Parâmetro | Valor |
| --------- | ----- |
| Filtros   | 64    |
| Kernel    | 3     |
| Ativação  | ReLU  |

Responsável por identificar correlações espaciais entre landmarks presentes em um mesmo instante temporal.

---

### Primeira Camada LSTM

| Parâmetro        | Valor |
| ---------------- | ----- |
| Unidades         | 64    |
| Ativação         | ReLU  |
| Return Sequences | True  |
| Dropout          | 0.2   |

Responsável pela modelagem da dinâmica temporal de curto prazo.

---

### Segunda Camada LSTM

| Parâmetro        | Valor |
| ---------------- | ----- |
| Unidades         | 32    |
| Ativação         | ReLU  |
| Return Sequences | False |
| Dropout          | 0.2   |

Responsável pela consolidação da representação temporal.

---

### Camada Densa

| Parâmetro | Valor |
| --------- | ----- |
| Unidades  | 128   |
| Ativação  | ReLU  |
| Dropout   | 0.2   |

Executa a combinação final dos atributos aprendidos.

---

### Camada de Saída

| Parâmetro | Valor   |
| --------- | ------- |
| Unidades  | 10      |
| Ativação  | Softmax |

Produz a distribuição de probabilidade entre as classes disponíveis.

---

# 3. Dataset e Estratégia de Treinamento

O dataset sintético utilizado na prova de conceito foi descontinuado.

A versão atual do modelo é treinada sobre uma vasta gama de dados reais e amplos, utilizando todo o potencial de múltiplos corpus linguísticos e visuais, visando uma cobertura extensa e generalizada do vocabulário de LIBRAS. 

## Datasets Utilizados

O vocabulário suportado é dinâmico e engloba o agrupamento de diversas fontes, incluindo:
*   **Academy Dataset**
*   **Healthcare Corpus Dataset** (dados extraídos temporalmente baseados em metadados para garantir o contexto clínico correto)
*   **V-LIBRASIL Dataset**
*   **LIBRAS Photo Dataset**

Essa combinação garante que o modelo reconheça centenas de sinais do cotidiano e do contexto de saúde, extrapolando o limite restrito de 10 classes iniciais.

---

## Data Augmentation

Para aumentar a capacidade de generalização, são aplicadas transformações controladas nos vetores de landmarks:

* Pequenas rotações;
* Alterações de escala;
* Ruído gaussiano de baixa intensidade.

Essas transformações simulam variações naturais de execução sem comprometer o significado dos sinais.

---

## Early Stopping

O treinamento utiliza monitoramento da perda de validação.

### Configuração

| Parâmetro | Valor     |
| --------- | --------- |
| Paciência | 12 épocas |

Objetivos:

* Evitar overfitting;
* Reduzir tempo de treinamento;
* Melhorar capacidade de generalização.

---

# 4. Requisitos Não Funcionais

## Metas de Desempenho

| Métrica                | Meta     |
| ---------------------- | -------- |
| Latência de Inferência | ≤ 50 ms  |
| Taxa de Processamento  | ≥ 30 FPS |
| Acurácia de Validação  | ≥ 90%    |

---

## Estratégias de Otimização

### Latência

* Conversão para TensorFlow Lite;
* Quantização INT8;
* Execução local no navegador.

### Taxa de Quadros

* Uso de `requestAnimationFrame`;
* Processamento assíncrono;
* Utilização de Web Workers.

### Precisão

* MediaPipe Holistic;
* Normalização espacial;
* Modelagem temporal via LSTM.

---

# 5. Considerações Arquiteturais

A nova arquitetura elimina as principais limitações observadas na primeira versão do sistema.

As melhorias mais relevantes incluem:

* Rastreamento detalhado dos dedos;
* Invariância espacial em relação à câmera;
* Representação explícita do movimento;
* Melhor modelagem temporal;
* Inferência otimizada para Edge Computing.

A combinação entre MediaPipe Holistic, engenharia avançada de atributos e a arquitetura híbrida CNN-1D + LSTM estabelece uma base sólida para aplicações acadêmicas e profissionais de tradução automática de LIBRAS em tempo real.

---

# 6. Conclusão

A arquitetura proposta representa uma evolução significativa em relação ao protótipo inicial, fornecendo maior robustez, precisão e capacidade de generalização para um vocabulário dinâmico e abrangente.

A integração entre visão computacional avançada, processamento temporal e execução local permite atender simultaneamente aos requisitos de desempenho, privacidade e acessibilidade exigidos por sistemas modernos de tradução automática de LIBRAS.

Essa abordagem estabelece uma base tecnológica robusta em Edge Computing, pronta para ser escalada, processando múltiplos datasets através de paralelismo e viabilizando a inclusão social a partir da tecnologia.
