# Estratégia de Testes Automatizados e Garantia de Qualidade

**Projeto:** Tradutor de Libras para Áudio com Inteligência Artificial
**Escopo:** Validação do Pipeline de IA (MediaPipe Holistic + CNN-1D + LSTM) e Interface de Edge Computing
**Versão:** 2.0 (Arquitetura Híbrida)

---

# 1. Visão Geral e Filosofia de Testes

Este documento define a estratégia de testes e o plano de garantia de qualidade do sistema **Tradutor de Libras para Áudio**.

Com a evolução da arquitetura para utilização do **MediaPipe Holistic** (53 landmarks e 159 atributos por frame) e da rede neural híbrida **CNN-1D + LSTM**, a infraestrutura de testes foi reformulada para assegurar que:

1. A normalização espacial invariante baseada na anatomia do usuário seja matematicamente correta.
2. O buffer circular de tempo real manipule adequadamente tensores com dimensionalidade `(30, 159)`.
3. O fenômeno de *mode collapse* seja monitorado e mitigado por meio de validações rigorosas das distribuições de probabilidade.
4. O sistema opere integralmente em ambiente local (*Edge Computing*), mantendo latência inferior a **50 ms**.

---

# 2. Pirâmide de Testes e Stack Tecnológica

O projeto adota a Pirâmide de Testes tradicional, adaptada para aplicações de Inteligência Artificial executadas diretamente no navegador.

```text
       /\
      /  \      Testes E2E (5–10%) → Playwright
     /----\
    /      \    Testes de Componentes (30–40%) → React Testing Library + Happy DOM
   /--------\
  /          \  Testes Unitários (50–60%) → Vitest
 /------------\
```

## 2.1 Ferramentas Utilizadas

### Testes Unitários — Vitest

Escolhido pela alta performance e integração nativa com o ecossistema Vite.

### Ambiente DOM — Happy DOM

Alternativa leve ao JSDOM para simulação de APIs do navegador.

### Testes de Componentes — React Testing Library (RTL)

Focada na validação do comportamento observado pelo usuário, evitando dependência de detalhes internos de implementação.

### Testes End-to-End (E2E) — Playwright

Permite validar fluxos completos em navegadores reais (Chromium, Firefox e WebKit), incluindo simulação de dispositivos e captura de mídia.

---

# 3. Metas de Cobertura de Código

A cobertura de testes é utilizada como critério obrigatório de aprovação no pipeline de Integração Contínua (CI).

| Métrica                 | Cobertura Mínima | Objetivo                                           |
| ----------------------- | ---------------- | -------------------------------------------------- |
| Linhas (Lines)          | ≥ 90%            | Garantir execução da maior parte do fluxo lógico   |
| Instruções (Statements) | ≥ 90%            | Validar expressões matemáticas e regras de negócio |
| Funções (Functions)     | ≥ 90%            | Evitar funções não testadas                        |
| Ramificações (Branches) | ≥ 85%            | Cobrir decisões críticas do sistema                |

Qualquer alteração que reduza os indicadores abaixo desses limites deverá falhar automaticamente no pipeline de CI.

---

# 4. Testes Unitários (Vitest)

Os testes unitários concentram-se na lógica de negócio desacoplada da interface gráfica, principalmente no módulo:

```text
client/src/lib/libras.ts
```

## 4.1 Processamento Espacial e Temporal

### UN-01 — Normalização por Origem Anatômica

**Objetivo**

Validar que o ponto médio entre os ombros (landmarks 11 e 12) seja corretamente definido como origem do sistema de coordenadas.

**Entrada**

* Vetor contendo 53 landmarks com coordenadas conhecidas.

**Critérios de Aceitação**

* O ponto médio dos ombros deve ser transformado para `(0,0,0)`.
* As distâncias relativas entre articulações devem permanecer inalteradas.

---

### UN-02 — Cálculo de Deltas Temporais

**Objetivo**

Garantir a precisão do cálculo de velocidade instantânea entre frames consecutivos.

**Entrada**

* Duas matrizes de landmarks com deslocamento linear controlado.

**Critérios de Aceitação**

* O vetor resultante deve conter os valores corretos de `Δx`, `Δy` e `Δz`.

---

### Exemplo de Implementação

```typescript
import { describe, it, expect } from 'vitest';
import {
  normalizeSpatialInvariance,
  calculateTemporalDeltas
} from './libras';

describe('Módulo de Engenharia de Atributos de LIBRAS', () => {
  describe('normalizeSpatialInvariance', () => {
    it('deve utilizar o centro dos ombros como origem', () => {
      const mockRawLandmarks = new Float32Array(53 * 3);

      mockRawLandmarks[11 * 3] = 0.4;
      mockRawLandmarks[11 * 3 + 1] = 0.5;

      mockRawLandmarks[12 * 3] = 0.6;
      mockRawLandmarks[12 * 3 + 1] = 0.5;

      mockRawLandmarks[20 * 3] = 0.8;
      mockRawLandmarks[20 * 3 + 1] = 0.9;
      mockRawLandmarks[20 * 3 + 2] = 0.1;

      const normalized =
        normalizeSpatialInvariance(mockRawLandmarks);

      expect(normalized[20 * 3]).toBeCloseTo(0.3);
      expect(normalized[20 * 3 + 1]).toBeCloseTo(0.4);
      expect(normalized[20 * 3 + 2]).toBeCloseTo(0.1);
    });
  });

  describe('Buffer Circular da CNN-1D', () => {
    it('deve preservar a estrutura temporal de 159 atributos', () => {
      const buffer: number[][] = [];
      const frameFeatures = new Array(159).fill(0.5);

      for (let i = 0; i < 35; i++) {
        buffer.push(frameFeatures);

        if (buffer.length > 30) {
          buffer.shift();
        }
      }

      expect(buffer.length).toBe(30);
      expect(buffer[0].length).toBe(159);
    });
  });
});
```

---

# 5. Testes de Componentes (React Testing Library)

Os testes de componentes validam estados visuais, comportamento dos hooks e integração entre UI e lógica de negócio.

## 5.1 Componente `LibrasTranslator.tsx`

### Estado Inicial

Validar:

* Carregamento dos arquivos:

  * `libras_model.tflite`
  * `class_mapping.json`
  * `scaler.json`
* Bloqueio do botão **Iniciar Câmara** até a conclusão do carregamento do modelo.

### Fluxo Positivo de Inferência

Simular uma previsão com:

```text
confiança > 0.70
```

**Resultado esperado**

* Exibição imediata do texto reconhecido.
* Exemplo: `"Obrigado"`.

### Fluxos de Exceção

#### Falha de Permissão da Câmera

Validar que:

* O erro seja tratado.
* Uma mensagem amigável seja exibida ao usuário.

#### Falha no Carregamento do Modelo

Validar que:

* O usuário seja informado sobre a indisponibilidade temporária do sistema.

---

## 5.2 Componente `Home.tsx`

A página inicial deve refletir corretamente a arquitetura técnica atual.

```typescript
import { render, screen } from '@testing-library/react';
import Home from './Home';

test('deve exibir a documentação técnica atualizada', () => {
  render(<Home />);

  expect(
    screen.getByText(/MediaPipe Holistic/i)
  ).toBeInTheDocument();

  expect(
    screen.getByText(/53 landmarks/i)
  ).toBeInTheDocument();

  expect(
    screen.getByText(/CNN-1D \+ LSTM/i)
  ).toBeInTheDocument();

  expect(
    screen.getByText(/Invariância Espacial/i)
  ).toBeInTheDocument();
});
```

---

# 6. Testes End-to-End (Playwright)

Os testes E2E validam a integração completa entre:

* Webcam
* Pipeline de visão computacional
* TensorFlow Lite
* Interface gráfica
* Síntese de voz

## 6.1 Cenários Críticos

### E2E-01 — Tradução Completa com Síntese de Voz

**Fluxo**

1. Usuário acessa `/translator`.
2. Permissão da câmera é concedida.
3. O Playwright injeta um fluxo de vídeo sintético.
4. O modelo retorna:

   * Classe: `"Água"`
   * Confiança: `85%`

**Resultado Esperado**

* Atualização do elemento:

```html
<div data-testid="prediction-result">
```

* Exibição da palavra **Água**.
* Chamada da API:

```javascript
window.speechSynthesis.speak(...)
```

com:

```javascript
{
  lang: 'pt-BR',
  rate: 0.9
}
```

---

### E2E-02 — Mitigação de Mode Collapse

**Fluxo**

* Injetar previsões instáveis com confiança máxima de 55%.

**Resultado Esperado**

* Nenhum áudio é reproduzido.
* Nenhum texto é atualizado.
* O limiar de confiança impede emissões incorretas.

---

## 6.2 Configuração do Ambiente E2E

```typescript
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',

  use: {
    headless: true,

    launchOptions: {
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        '--use-file-for-fake-video-capture=e2e/assets/libras_test_sequence.y4m'
      ]
    }
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] }
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] }
    }
  ]
});
```

---

# 7. Pipeline de Integração Contínua (CI/CD)

O pipeline automatizado atua como mecanismo de controle de qualidade e aprovação de mudanças.

```text
GitHub Actions
      │
      ▼
1. Instalação
   pnpm install

      │
      ▼
2. Lint e Build
   pnpm lint && pnpm build

      │
      ▼
3. Testes Unitários
   pnpm test:coverage

      │
      ▼
4. Testes E2E
   pnpm test:e2e

      │
      ▼
Deploy Autorizado (Vercel)
```

---

# 8. Matriz de Rastreabilidade

| ID     | Tipo          | Validador          | Métrica                       | Resultado Esperado |
| ------ | ------------- | ------------------ | ----------------------------- | ------------------ |
| RNF-01 | Não Funcional | Performance Hook   | Inferência ≤ 50 ms            | ✅                  |
| RNF-02 | Não Funcional | Cobertura Vitest   | Branches ≥ 85%                | ✅                  |
| RF-01  | Funcional     | RTL / Web Speech   | Áudio após confiança ≥ 70%    | ✅                  |
| RF-02  | Funcional     | Testes Geométricos | Invariância espacial validada | ✅                  |

---

## Considerações Finais

Este documento constitui a referência oficial de qualidade do projeto. Qualquer alteração na arquitetura de rede neural, no pipeline de pré-processamento ou na representação dos landmarks deverá ser acompanhada pela atualização desta estratégia e de suas respectivas suítes de testes.

O objetivo é garantir a confiabilidade, a precisão e a robustez operacional do tradutor de Libras em ambientes de Edge Computing, mantendo padrões elevados de qualidade durante todo o ciclo de desenvolvimento.
