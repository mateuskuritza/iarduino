# Projeto: Reconhecimento de Itens com Arduino e Visão Computacional

Este projeto integra um sistema de reconhecimento de imagens utilizando Python, OpenCV e TensorFlow com um Arduino, permitindo acender LEDs correspondentes ao item reconhecido pela câmera.

## Funcionalidades
- Captura de imagens de diferentes itens para treinamento.
- Treinamento de um modelo de classificação de imagens.
- Interface gráfica para gerenciar itens, treinar modelos e iniciar a detecção.
- Comunicação serial entre Python e Arduino para acionar LEDs conforme o item reconhecido.
- Simulação do circuito no Wokwi.

## Passo a Passo para Rodar o Projeto

### Pré requisitos
- Python3 instalado
- Ambiente virtual configurado para a instalação das dependências (a versão do Python precisa ser entre 2.8 e 3.12)
```bash
python -m venv .venv
source .venv/bin/activate
```

### 1. Instale as dependências Python
```bash
pip install -r requirements.txt
```

### 2. Capture imagens para cada item
```bash
python src/capture.py <nome_item>
```
Repita para cada item que deseja cadastrar.

### 3. Pré-processe as imagens (opcional, feito automaticamente ao treinar)
```bash
python src/pre_process.py
```

### 4. Treine o modelo
```bash
python src/train_model.py
```

### 5. Compile e envie o código para o Arduino
Abra `arduino/main/main.ino` na IDE do Arduino e envie para sua placa (ex: Arduino Uno ou Mega).

### 6. Execute a interface gráfica
```bash
python src/main.py
```

- Use a interface para adicionar/remover itens, treinar modelos e iniciar a detecção.
- Ao iniciar a detecção, informe a porta serial do Arduino (ex: `/dev/ttyACM0` no Linux/Mac ou `COM3` no Windows).

### 7. Simule o circuito (opcional)
Abra `diagram.json` no [Wokwi](https://wokwi.com/projects/426892168473742337) para simular o circuito.

---