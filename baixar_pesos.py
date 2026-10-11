"""Roda na montagem da imagem: baixa os pesos do pacote pt-BR e a voz padrão."""
import ptbr

if __name__ == "__main__":
    print("checkpoint:", ptbr.montar_checkpoint(), flush=True)
    print("voz padrão:", ptbr.baixar_voz_padrao(), flush=True)
