import yfinance as yf
import os
import csv

# Windows/OneDrive silently mangles a filename whose name (before the first
# '.') matches one of these reserved device names -- e.g. 'CON.DE.csv'
# (Continental AG, part of the DAX universe) was found on disk as
# '_ON.DE.csv'. Prefixing with '_' sidesteps the collision.
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Exchange suffixes actually used by this project's universes (NSE/BSE, DAX,
# LSE, HSI, Nikkei, KOSPI) plus a few common extras, so a dot-suffix that
# ISN'T one of these is treated as a US share-class separator instead (see
# normalize_ticker_for_download).
KNOWN_EXCHANGE_SUFFIXES = {"NS", "BO", "DE", "L", "HK", "PA", "MI", "TO", "AX", "SW", "T", "KS"}


def normalize_ticker_for_download(symbol):
    """
    Yahoo Finance expects a hyphen for US share classes (BRK-B, BF-B), while
    the source universe CSVs use a dot (BRK.B, BF.B) -- the Wikipedia
    convention. Foreign tickers also use a dot, but for an exchange suffix
    (ADANIENT.NS, AIR.DE, AAF.L) that must NOT be touched, so only a
    dot-suffix that isn't a known exchange code gets converted to a hyphen.

    Some LSE tickers in this project's universe CSVs use a '/' for two
    different things, neither of which Yahoo's API accepts as-is:
      - A trailing EPIC dot before the exchange suffix (e.g. 'AV/.L' for
        Aviva's EPIC 'AV.', 'BP/.L' for BP's EPIC 'BP.') -- Yahoo's real
        ticker just drops it entirely ('AV.L', 'BP.L').
      - A genuine share-class separator (e.g. 'BT/A.L' for BT Group's A
        shares) -- Yahoo uses a hyphen, same as the US convention
        ('BT-A.L').
    """
    if "/" in symbol:
        prefix, rest = symbol.split("/", 1)
        if rest.startswith("."):
            symbol = prefix + rest
        else:
            symbol = prefix + "-" + rest

    if "." in symbol:
        prefix, suffix = symbol.rsplit(".", 1)
        if suffix.upper() not in KNOWN_EXCHANGE_SUFFIXES and suffix.isalpha() and len(suffix) <= 2:
            return f"{prefix}-{suffix}"
    return symbol


def sanitize_filename(symbol):
    """
    Make a symbol safe to use as a filename on disk.
      - '/' is a path separator, not a valid filename character -- some LSE
        dual-class tickers contain one (e.g. 'RR/.L', 'BT/A.L') and were
        silently failing to save at all.
      - A name matching a Windows-reserved device name gets mangled by
        Windows/OneDrive on sync (see WINDOWS_RESERVED_NAMES above).

    The *original* symbol is preserved separately in a sidecar map file so
    downstream code can recover the true ticker even though the file on
    disk may have a different, sanitised name.
    """
    safe = symbol.replace("/", "_")

    name_part = safe.split(".", 1)[0]
    if name_part.upper() in WINDOWS_RESERVED_NAMES:
        safe = "_" + safe

    return safe


def download_stock_data(stock_list, start_date, end_date, save_dir = "data"):
    os.makedirs(save_dir, exist_ok=True)
    successful = 0
    failed = 0

    symbol_map_path = os.path.join(save_dir, "_symbol_filename_map.csv")
    symbol_map_rows = []

    for symbol in stock_list:
        print(f"Downlaoding...{symbol}")

        download_symbol = normalize_ticker_for_download(symbol)
        safe_name = sanitize_filename(symbol)
        file_path = os.path.join(save_dir, f"{safe_name}.csv")

        try:
             df = yf.download(download_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False, multi_level_index=False)
             if df.empty:
                 print(f"Skipping {symbol} (No Data)")
                 failed += 1
                 continue

             df.to_csv(file_path)
             print(f"Saved -> {file_path}")
             symbol_map_rows.append((symbol, safe_name))

             successful += 1

        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            failed += 1

    if symbol_map_rows:
        with open(symbol_map_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Symbol", "Filename"])
            writer.writerows(symbol_map_rows)

    print("Download_Summary: ")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
