MODEL_NAMES = ["vadclip", "corevad", "anyanomaly"]


def get_model_class(name):
    # 各モデルは依存ライブラリが異なる（別環境）ため、使うものだけ import する
    if name == "vadclip":
        from models.vadclip import VadCLIP
        return VadCLIP
    if name == "corevad":
        from models.corevad import CoReVAD
        return CoReVAD
    if name == "anyanomaly":
        from models.anyanomaly import AnyAnomaly
        return AnyAnomaly
    raise ValueError(f"未知のモデル: {name}（{MODEL_NAMES} から選択）")
