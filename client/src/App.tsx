import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LibrasTranslator from './components/LibrasTranslator';
import Home from './components/Home';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/translator" element={<LibrasTranslator />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
