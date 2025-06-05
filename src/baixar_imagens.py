from bing_image_downloader import downloader

# Lista de frutas e pastas
frutas = [
    # ("fruta banana com fundo branco", "data/captures/banana"),
    # ("fruta morango com fundo branco", "data/captures/morango"),
    # ("fruta limao com fundo branco", "data/captures/limao"),
    # ("copo de vidro com fundo branco", "data/captures/copo"),
    # ("rosto humano com fundo branco", "data/captures/rosto"),
    # ("xicara com fundo branco", "data/captures/xicara"),
    # ("prato com fundo branco", "data/captures/prato"),
    # ("garfo com fundo branco", "data/captures/garfo"),
    ("panela com fundo branco", "data/captures/panela"),
]

for fruta, pasta in frutas:
    downloader.download(
        fruta,
        limit=15,
        output_dir="data/captures",
        adult_filter_off=True,
        force_replace=False,
        timeout=30,
        verbose=True,
    )
    # Renomeia a pasta criada para o nome correto (sem espaços, acentos)
    import os, shutil

    nome_bing = (
        fruta.replace("ç", "c")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace(" ", "_")
    )
    pasta_bing = f"data/captures/{nome_bing}"
    if pasta_bing != pasta and os.path.exists(pasta_bing):
        if os.path.exists(pasta):
            shutil.rmtree(pasta)
        os.rename(pasta_bing, pasta)
