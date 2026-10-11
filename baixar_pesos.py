"""Roda na montagem da imagem: baixa os pesos do pacote pt-BR, o que o tokenizador busca na internet e a voz padrão."""
import ptbr

if __name__ == "__main__":
    checkpoint = ptbr.montar_checkpoint()
    print("checkpoint:", checkpoint, flush=True)
    ptbr.carregar_tokenizador(checkpoint)
    print("tokenizador pronto (segmentador chinês e mapa Cangjie baixados)", flush=True)
    print("voz padrão:", ptbr.baixar_voz_padrao(), flush=True)
