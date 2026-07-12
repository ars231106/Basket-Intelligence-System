from src.download_data import normalize_ticker_for_download

def test_korean_ticker_suffix_untouched():
    assert normalize_ticker_for_download("000080.KS") == "000080.KS"

def test_indian_nse_suffix_untouched():
    assert normalize_ticker_for_download("RELIANCE.NS") == "RELIANCE.NS"

def test_german_suffix_untouched():
    assert normalize_ticker_for_download("SAP.DE") == "SAP.DE"

def test_japanese_suffix_untouched():
    assert normalize_ticker_for_download("7203.T") == "7203.T"

def test_us_share_class_dot_convereted_to_hyphen():
    assert normalize_ticker_for_download("BRK.B") == "BRK-B"

def test_lse_share_class_slash_converted_to_hyphen():
    assert normalize_ticker_for_download("BT/A.L") == "BT-A.L"

def test_lse_epic_dot_slash_dropped():
    assert normalize_ticker_for_download("AV/.L") == "AV.L"