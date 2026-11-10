# Projeto X

Projeto para detecção e rastreio de alvos com YOLO, com controle de movimento via serial para hardware externo.

## Visão geral

- Leitura de tela via `mss`
- Detecção por `ultralytics`/YOLO
- Seleção do alvo mais relevante com persistência de tracking
- Envio de comandos serial para controle de hardware
- Visualização opcional com `--preview`

## Requisitos

- Python 3.10+
- Dependências do arquivo `limpo.py`
- Modelos treinados em `models/`

## Modelos

Os arquivos de modelo presentes na pasta `models/` fazem parte do repositório e podem ser enviados junto com o projeto.

## Execução

```bash
python limpo.py --preview
```

> Os artefatos pesados de modelo (`.pt`, `.onnx`) foram ignorados no Git para evitar uploads muito grandes.

## Observações

Este projeto foi estruturado para uso local e pode exigir ajustes conforme o hardware e a janela alvo.
