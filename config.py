# config.py
# Konfigurasi ID Google Drive dan parameter model

DRIVE_IDS = {
    # Folder model IndoBERT (fine-tuned)
    "model_indobert"    : "1kF74xhsbfFDlOcu4_DgwMBjYVSOuYSMd",

    # File leksikon
    "kbbi"              : "1RooLxbgIn3LN3nJgl6C5325F1Ciwugio",
    "kata_inggris"      : "11I2tLBZCUDY85HWelRHXCuWClnDS14Cj",
    "kata_serapan"      : "1g_TCVwcFfDLYYnob5Ylnm5LaxSDNtCmO",
    "akronim"           : "1TrHDSwcMCguu_UdEn7_hkXLytNNvTzMd",
    "daftar_lembaga"    : "1RfRHN6rTa-JPPXV0SCGq1nl2k0EUDRUJ",
    "daftar_nama_orang" : "1Kfehgx-OyfR1ePUbUq0Rzw96-zbvnBxT",
    "istilah_islam"     : "1JhbRw7-J7B6xyJ9GdFSyea1tjEplEEBB",
    "sample_correct_2025": "1gT7UwwHE7on1-kfJfIx2GQS61U8zH-eW",
}

# Konfigurasi model IndoBERT
MODEL_CFG = {
    "max_length" : 128,
    "num_labels" : 2,
}

# Konfigurasi Jaro-Winkler
JW_CFG = {
    "threshold" : 0.99,
    "top_k"     : 5,
    "p_factor"  : 0.1,
}
