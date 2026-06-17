import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LibrasTranslator from './components/LibrasTranslator';
import Home from './components/Home';
import './index.css';

/**
 * Componente Raiz da Aplicação.
 * 
 * Responsável por configurar o roteamento (React Router) do Frontend.
 * Define as rotas principais:
 * - "/" : Renderiza a Landing Page (Home)
 * - "/translator" : Renderiza a interface do Tradutor de Libras em tempo real.
 */
function App() {
  return (
    // Engloba a aplicação no contexto do roteador
    <BrowserRouter>
      <Routes>
        {/* Rota para a página de introdução/landing page */}
        <Route path="/" element={<Home />} />
        
        {/* Rota para o tradutor interativo contendo a câmera e o modelo IA */}
        <Route path="/translator" element={<LibrasTranslator />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
