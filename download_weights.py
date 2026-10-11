"""Runs at image build time: downloads the pt-BR pack weights, the data the tokenizer fetches online, and the default voice."""
import ptbr_model

if __name__ == "__main__":
    checkpoint = ptbr_model.assemble_checkpoint()
    print("checkpoint:", checkpoint, flush=True)
    ptbr_model.load_tokenizer(checkpoint)
    print("tokenizer ready (Chinese segmenter and Cangjie map downloaded)", flush=True)
    print("default voice:", ptbr_model.download_default_voice(), flush=True)
