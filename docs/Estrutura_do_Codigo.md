# Estrutura e Funcionalidade do Código — Tradutor de Libras

Este documento apresenta uma visão detalhada, porém concisa, da funcionalidade de todos os arquivos de código-fonte presentes no projeto, organizados por suas respectivas pastas. 

---

## 📁 `client/` (Frontend React)
Contém a aplicação web responsável por executar o modelo em tempo real no navegador do usuário utilizando Edge Computing.

### `client/src/`
- **`main.tsx`**: Ponto de entrada da aplicação React. Monta o componente raiz na DOM.
- **`App.tsx`**: Componente raiz que configura o roteamento da aplicação (usando `react-router-dom`), conectando a página inicial ao tradutor.
- **`App.css` / `index.css`**: Arquivos de estilização global contendo variáveis de design (cores, temas) e estilos gerais.
- **`setupTests.ts`**: Configurações de ambiente para testes unitários usando o framework Vitest / Jest.

### `client/src/components/`
- **`Home.tsx`**: Componente da página inicial (Landing Page). Apresenta a interface introduzindo os benefícios do tradutor (baixa latência, privacidade) com atalho para a tradução.
- **`Home.css`**: Estilos específicos aplicados exclusivamente à página inicial.
- **`LibrasTranslator.tsx`**: O núcleo da aplicação web. Gerencia a captura de vídeo (Webcam), extrai características via MediaPipe Holistic, acumula os frames em um buffer e realiza a inferência do sinal usando TensorFlow Lite. Também contém a lógica de "Text-to-Speech" (Web Speech API) controlada interativamente pelo usuário.
- **`LibrasTranslator.test.tsx`**: Casos de teste para validar a renderização e o comportamento do tradutor.

### `client/src/lib/`
- **`libras.ts`**: Arquivo de funções utilitárias matemáticas. Contém a lógica de "Engenharia de Atributos" para o frontend: recebe os landmarks do MediaPipe, calcula o centro geométrico (entre os ombros) e realiza a normalização espacial para gerar os 159 atributos esperados pelo modelo TFLite.
- **`libras.test.ts`**: Testes automatizados para garantir que a normalização espacial matemática em `libras.ts` ocorra corretamente.

---

## 📁 `scripts/` (Backend e Machine Learning)
Contém os scripts em Python utilizados para processamento dos datasets, extração de características (landmarks) e treinamento da rede neural.

### Extração e Processamento de Dados
- **`extract_features.py`**: Processa vídeos do *V-LIBRASIL Dataset*. Usa o MediaPipe Holistic para extrair 159 atributos de cada frame (tronco e mãos), aplica a normalização espacial e exporta as sequências temporais de matrizes para arquivos `.npy`. Otimizado com processamento paralelo (multiprocessing).
- **`extract_landmarks.py`**: Script de uso geral (genérico) para varrer e extrair landmarks de coleções completas de vídeos em subdiretórios, padronizando-os em janelas de 30 frames.
- **`extract_features_photos.py`**: Adaptação do pipeline de extração de características focado em bases de dados compostas por fotos estáticas (como o *LIBRAS Photo Dataset*).
- **`process_healthcare_corpus.py`**: Script de processamento sob medida para o *Healthcare Corpus Dataset*, que recorta instantes específicos dos vídeos (clipes temporais) baseado em anotações clínicas antes da extração.
- **`augment_data.py`**: Aplica técnicas de *Data Augmentation* nas matrizes `.npy` já extraídas. Adiciona pequenas rotações, zoom in/out (escala) e ruído gaussiano aos marcos tridimensionais, aumentando artificialmente a robustez e generalização do modelo.

### Treinamento e Exportação
- **`train_model.py`**: O núcleo do processo de *Machine Learning*. Lê todos os `.npy` extraídos e aumentados, constrói a arquitetura híbrida **CNN-1D + Bidirectional LSTM** (com camadas de Dropout para evitar overfitting) e treina o modelo Keras salvando o melhor resultado no formato `.h5`.
- **`export_tflite.py`**: Converte o modelo treinado `.h5` do Keras para a versão leve e portátil do TensorFlow Lite (`.tflite`). Esse script aplica quantizações quando necessário para reduzir a latência de inferência no navegador.
