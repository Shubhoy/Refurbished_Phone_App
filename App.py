from __future__ import annotations
import streamlit as st
import sqlite3
import bcrypt
import pandas as pd

import sqlite3
import os

DB_FILE = "phones.db"

PHONES_SCHEMA = """
CREATE TABLE IF NOT EXISTS phones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand TEXT NOT NULL,
    model TEXT NOT NULL
);
"""

PRICES_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_id INTEGER,
    price INTEGER,
    FOREIGN KEY (phone_id) REFERENCES phones(id)
);
"""

LISTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_id INTEGER,
    seller TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (phone_id) REFERENCES phones(id)
);
"""

def ensure_schema():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.executescript(PHONES_SCHEMA + PRICES_SCHEMA + LISTINGS_SCHEMA)
    conn.commit()
    conn.close()

# Run this at startup
ensure_schema()

# --- DB bootstrap (PASTE THIS JUST BELOW YOUR IMPORTS) ---
import sqlite3

DB_PATH = "refurb_phones.db"  # <- use this single file name everywhere

PHONES_SCHEMA = """
CREATE TABLE IF NOT EXISTS phones (
    id INTEGER PRIMARY KEY,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    storage TEXT DEFAULT '',
    color TEXT DEFAULT '',
    condition TEXT CHECK(condition IN ('New','Good','Scrap')) NOT NULL DEFAULT 'Good',
    cost_price REAL NOT NULL DEFAULT 0,
    base_net REAL NOT NULL DEFAULT 0,
    stock INTEGER NOT NULL DEFAULT 0,
    reserved_b2b INTEGER NOT NULL DEFAULT 0,
    discontinued INTEGER NOT NULL DEFAULT 0,
    tags TEXT DEFAULT ''
);
"""

PRICES_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    phone_id INTEGER,
    platform TEXT,
    auto_price REAL,
    manual_override REAL,
    PRIMARY KEY (phone_id, platform)
);
"""

LISTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    phone_id INTEGER,
    platform TEXT,
    status TEXT,
    reason TEXT,
    PRIMARY KEY (phone_id, platform)
);
"""

def ensure_schema():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # Create tables if they don't exist
    cur.executescript(PHONES_SCHEMA + PRICES_SCHEMA + LISTINGS_SCHEMA)

    # If an older phones table exists, add any missing columns
    cur.execute("PRAGMA table_info(phones)")
    existing = {row[1] for row in cur.fetchall()}
    need_cols = {
        "storage": "TEXT DEFAULT ''",
        "color": "TEXT DEFAULT ''",
        "condition": "TEXT DEFAULT 'Good'",
        "cost_price": "REAL NOT NULL DEFAULT 0",
        "base_net": "REAL NOT NULL DEFAULT 0",
        "stock": "INTEGER NOT NULL DEFAULT 0",
        "reserved_b2b": "INTEGER NOT NULL DEFAULT 0",
        "discontinued": "INTEGER NOT NULL DEFAULT 0",
        "tags": "TEXT DEFAULT ''",
    }
    for col, ddl in need_cols.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE phones ADD COLUMN {col} {ddl}")

    # Seed a few rows if the table is empty (helps you see the UI working)
    cur.execute("SELECT COUNT(*) FROM phones")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO phones(brand, model, condition, storage, color, cost_price, base_net, stock, reserved_b2b, tags) VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                ("Apple","iPhone 12","Good","128GB","Black",250,320,5,1,"hot"),
                ("Samsung","Galaxy S21","New","256GB","Violet",300,380,3,0,"flagship"),
                ("Xiaomi","Redmi Note 10","Scrap","64GB","Green",40,70,0,0,"repair"),
            ],
        )

    con.commit()
    con.close()

# 🚨 VERY IMPORTANT: call this BEFORE any DB SELECT/INSERT in your script
ensure_schema()
# --- end bootstrap ---

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("INSERT INTO phones (brand, model) VALUES (?, ?)", ("Apple", "iPhone 13"))
cur.execute("INSERT INTO phones (brand, model) VALUES (?, ?)", ("Samsung", "Galaxy S21"))
conn.commit()
conn.close()
con = sqlite3.connect("database.db")
cur = con.cursor()
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

DB_PATH = "refurb_phones.db"

# ---------- Configuration ----------
PLATFORMS = ["X", "Y", "Z"]
FEE_RULES = {
    "X": {"percent": 0.10, "fixed": 0.0},        # 10%
    "Y": {"percent": 0.08, "fixed": 2.0},        # 8% + 2
    "Z": {"percent": 0.12, "fixed": 0.0},        # 12%
}

# Internal canonical conditions
CONDITIONS = ["New", "Good", "Scrap"]

# Mapping to platform categories
# Note: Z does NOT support "Scrap" (will be treated as unsupported)
COND_MAP = {
    "X": {"New": "New", "Good": "Good", "Scrap": "Scrap"},
    "Y": {"New": "3 stars (Excellent)", "Good": "2 stars (Good)", "Scrap": "1 star (Usable)"},
    "Z": {"New": "New", "Good": "Good"},  # 'Scrap' unsupported on Z
}

# ---------- Utilities ----------

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                pwdhash BLOB
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS phones (
                id INTEGER PRIMARY KEY,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                storage TEXT,
                color TEXT,
                condition TEXT CHECK(condition IN ('New','Good','Scrap')) NOT NULL,
                cost_price REAL NOT NULL DEFAULT 0,
                base_net REAL NOT NULL DEFAULT 0,     -- desired net after platform fees
                stock INTEGER NOT NULL DEFAULT 0,
                reserved_b2b INTEGER NOT NULL DEFAULT 0, -- units reserved (not listable)
                discontinued INTEGER NOT NULL DEFAULT 0, -- boolean 0/1
                tags TEXT DEFAULT ''
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                phone_id INTEGER,
                platform TEXT,
                auto_price REAL,       -- auto computed from base_net + fees
                manual_override REAL,  -- nullable; overrides auto_price if not null
                PRIMARY KEY (phone_id, platform),
                FOREIGN KEY(phone_id) REFERENCES phones(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS listings (
                phone_id INTEGER,
                platform TEXT,
                status TEXT,          -- 'listed' | 'failed' | 'not_listed'
                reason TEXT,          -- failure reason if any
                PRIMARY KEY (phone_id, platform),
                FOREIGN KEY(phone_id) REFERENCES phones(id)
            )
            """
        )
        con.commit()


def seed_admin():
    """Create a default admin user (admin / admin123) if none exists."""
    with get_conn() as con:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        (count,) = cur.fetchone()
        if count == 0:
            username = "admin"
            pwd = "admin123".encode()
            salt = bcrypt.gensalt()
            h = bcrypt.hashpw(pwd, salt)
            cur.execute("INSERT INTO users(username, pwdhash) VALUES(?,?)", (username, h))
            con.commit()


@dataclass
class PriceCalcResult:
    platform: str
    price: float
    percent: float
    fixed: float


def fee_components(platform: str) -> Tuple[float, float]:
    rule = FEE_RULES[platform]
    return rule["percent"], rule["fixed"]


def compute_listing_price(base_net: float, platform: str) -> PriceCalcResult:
    """Given desired net proceeds (base_net), compute the list price to achieve it.
    price = (base_net + fixed) / (1 - percent)
    """
    p, f = fee_components(platform)
    denom = (1.0 - p)
    list_price = (base_net + f) / denom if denom > 0 else float("inf")
    return PriceCalcResult(platform, round(list_price, 2), p, f)


def compute_profit(list_price: float, cost_price: float, platform: str) -> float:
    p, f = fee_components(platform)
    net = list_price * (1 - p) - f
    return round(net - cost_price, 2)


def ensure_prices_rows(phone_id: int):
    with get_conn() as con:
        cur = con.cursor()
        for plat in PLATFORMS:
            cur.execute(
                "INSERT OR IGNORE INTO prices(phone_id, platform, auto_price, manual_override) VALUES(?,?,NULL,NULL)",
                (phone_id, plat),
            )
        con.commit()


# ---------- Auth ----------

def login_form() -> bool:
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username", value="admin")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Sign in"):
        with get_conn() as con:
            cur = con.cursor()
            cur.execute("SELECT pwdhash FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            if row and bcrypt.checkpw(password.encode(), row[0]):
                st.session_state["auth"] = True
                st.session_state["user"] = username
                st.sidebar.success("Authenticated")
                return True
            else:
                st.sidebar.error("Invalid credentials")
                st.session_state["auth"] = False
                return False
    return st.session_state.get("auth", False)


# ---------- UI Helpers ----------

def phone_to_series(row: sqlite3.Row) -> Dict:
    return {
        "id": row[0],
        "brand": row[1],
        "model": row[2],
        "storage": row[3],
        "color": row[4],
        "condition": row[5],
        "cost_price": row[6],
        "base_net": row[7],
        "stock": row[8],
        "reserved_b2b": row[9],
        "discontinued": bool(row[10]),
        "tags": row[11] or "",
    }


def validate_phone_input(data: Dict) -> Optional[str]:
    # Basic input validation & sanitization
    required = ["brand", "model", "condition"]
    for r in required:
        if not str(data.get(r, "")).strip():
            return f"Missing required field: {r}"
    if data["condition"] not in CONDITIONS:
        return "Invalid condition"
    try:
        if float(data.get("cost_price", 0)) < 0 or float(data.get("base_net", 0)) < 0:
            return "Prices cannot be negative"
        if int(data.get("stock", 0)) < 0 or int(data.get("reserved_b2b", 0)) < 0:
            return "Stock values cannot be negative"
    except Exception:
        return "Numeric fields must be valid numbers"
    return None


# ---------- Pages ----------

def page_inventory():
    st.header("📦 Inventory Management")
    with st.expander("➕ Add / Update Phone", expanded=False):
        with get_conn() as con:
            col1, col2, col3 = st.columns(3)
            brand = col1.text_input("Brand")
            model = col2.text_input("Model")
            condition = col3.selectbox("Condition", CONDITIONS)
            storage = col1.text_input("Storage (e.g., 128GB)")
            color = col2.text_input("Color")
            cost_price = col3.number_input("Cost Price", min_value=0.0, step=1.0)
            base_net = col1.number_input("Desired Net (after fees)", min_value=0.0, step=1.0)
            stock = col2.number_input("Stock Qty", min_value=0, step=1)
            reserved_b2b = col3.number_input("Reserved for B2B/Direct", min_value=0, step=1)
            discontinued = col1.checkbox("Discontinued?")
            tags = col2.text_input("Tags (comma-separated)")

            phone_id = st.text_input("(Optional) Phone ID to Update")
            c1, c2 = st.columns(2)
            if c1.button("Save"):
                data = {
                    "brand": brand.strip(),
                    "model": model.strip(),
                    "condition": condition,
                    "storage": storage.strip(),
                    "color": color.strip(),
                    "cost_price": cost_price,
                    "base_net": base_net,
                    "stock": stock,
                    "reserved_b2b": reserved_b2b,
                    "discontinued": int(discontinued),
                    "tags": tags.strip(),
                }
                err = validate_phone_input(data)
                if err:
                    st.error(err)
                else:
                    cur = con.cursor()
                    if phone_id:
                        try:
                            pid = int(phone_id)
                            cur.execute(
                                """
                                UPDATE phones SET brand=?, model=?, storage=?, color=?, condition=?, cost_price=?, base_net=?, stock=?, reserved_b2b=?, discontinued=?, tags=?
                                WHERE id=?
                                """,
                                (
                                    data["brand"], data["model"], data["storage"], data["color"], data["condition"],
                                    data["cost_price"], data["base_net"], data["stock"], data["reserved_b2b"],
                                    data["discontinued"], data["tags"], pid,
                                ),
                            )
                            con.commit()
                            ensure_prices_rows(pid)
                            st.success(f"Updated phone #{pid}")
                        except ValueError:
                            st.error("Phone ID must be an integer for updates")
                    else:
                        cur.execute(
                            """
                            INSERT INTO phones(brand, model, storage, color, condition, cost_price, base_net, stock, reserved_b2b, discontinued, tags)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                data["brand"], data["model"], data["storage"], data["color"], data["condition"],
                                data["cost_price"], data["base_net"], data["stock"], data["reserved_b2b"],
                                data["discontinued"], data["tags"],
                            ),
                        )
                        pid = cur.lastrowid
                        con.commit()
                        ensure_prices_rows(pid)
                        st.success(f"Added phone #{pid}")

            if c2.button("Delete by ID"):
                try:
                    pid = int(phone_id)
                    cur = con.cursor()
                    cur.execute("DELETE FROM phones WHERE id=?", (pid,))
                    cur.execute("DELETE FROM prices WHERE phone_id=?", (pid,))
                    cur.execute("DELETE FROM listings WHERE phone_id=?", (pid,))
                    con.commit()
                    st.warning(f"Deleted phone #{pid}")
                except Exception as e:
                    st.error(f"Delete failed: {e}")

    st.divider()
    st.subheader("📄 Bulk Upload (CSV)")
    st.caption("Columns: brand,model,condition,storage,color,cost_price,base_net,stock,reserved_b2b,discontinued,tags")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            required = {"brand","model","condition"}
            if not required.issubset(df.columns):
                st.error("Missing required columns in CSV")
            else:
                with get_conn() as con:
                    cur = con.cursor()
                    added = 0
                    for _, r in df.fillna("").iterrows():
                        data = {
                            "brand": str(r.get("brand","")),
                            "model": str(r.get("model","")),
                            "condition": str(r.get("condition","")),
                            "storage": str(r.get("storage","")),
                            "color": str(r.get("color","")),
                            "cost_price": float(r.get("cost_price", 0) or 0),
                            "base_net": float(r.get("base_net", 0) or 0),
                            "stock": int(r.get("stock", 0) or 0),
                            "reserved_b2b": int(r.get("reserved_b2b", 0) or 0),
                            "discontinued": int(r.get("discontinued", 0) or 0),
                            "tags": str(r.get("tags","")),
                        }
                        err = validate_phone_input(data)
                        if err:
                            st.warning(f"Row skipped ({data['brand']} {data['model']}): {err}")
                            continue
                        cur.execute(
                            """
                            INSERT INTO phones(brand, model, storage, color, condition, cost_price, base_net, stock, reserved_b2b, discontinued, tags)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                data["brand"], data["model"], data["storage"], data["color"], data["condition"],
                                data["cost_price"], data["base_net"], data["stock"], data["reserved_b2b"],
                                data["discontinued"], data["tags"],
                            ),
                        )
                        pid = cur.lastrowid
                        ensure_prices_rows(pid)
                        added += 1
                    con.commit()
                    st.success(f"Bulk upload complete. Added {added} phones.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("📚 Current Inventory")
    with get_conn() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM phones ORDER BY id DESC")
        rows = cur.fetchall()
        df = pd.DataFrame([phone_to_series(r) for r in rows])
        if df.empty:
            st.info("No phones yet. Add some above or via CSV.")
        else:
            st.dataframe(df, use_container_width=True)


def page_pricing():
    st.header("💰 Automated Price Updates & Overrides")
    st.caption("Auto-prices target your Desired Net (after fees). Manual overrides take precedence for listings.")
    with get_conn() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM phones ORDER BY id DESC")
        phones = cur.fetchall()
        if not phones:
            st.info("Add phones first in Inventory page.")
            return

        opts = [(r[0], f"#{r[0]} - {r['brand']} {r['model']} ({r['condition']})") for r in phones]
        sel = st.selectbox("Select Phone", options=opts, format_func=lambda x: x[1])
        pid = sel[0]
        phone = [p for p in phones if p[0] == pid][0]
        st.write(f"**Base Net target:** {phone['base_net']} | **Cost:** {phone['cost_price']}")

        # Compute auto prices
        autos: List[PriceCalcResult] = [compute_listing_price(phone['base_net'], plat) for plat in PLATFORMS]
        auto_df = pd.DataFrame([{
            "Platform": a.platform,
            "Auto Price": a.price,
            "Fee %": f"{int(a.percent*100)}%",
            "Fixed Fee": a.fixed,
            "Profit @ Auto": compute_profit(a.price, phone['cost_price'], a.platform)
        } for a in autos])
        st.dataframe(auto_df, use_container_width=True)

        # Load current overrides
        cur.execute("SELECT platform, auto_price, manual_override FROM prices WHERE phone_id=?", (pid,))
        price_rows = {r[0]: {"auto_price": r[1], "manual_override": r[2]} for r in cur.fetchall()}

        # Update auto prices button
        if st.button("Recompute & Save Auto Prices"):
            for a in autos:
                cur.execute(
                    "UPDATE prices SET auto_price=? WHERE phone_id=? AND platform=?",
                    (a.price, pid, a.platform),
                )
            con.commit()
            st.success("Auto prices updated.")

        st.subheader("Manual Overrides")
        for plat in PLATFORMS:
            current = price_rows.get(plat, {})
            current_override = current.get("manual_override")
            override = st.number_input(
                f"{plat} Override (blank = none)",
                value=float(current_override) if current_override is not None else 0.0,
                min_value=0.0,
                step=1.0,
                key=f"ovr_{plat}"
            )
            use_override = st.checkbox(
                f"Use override for {plat}?",
                value=current_override is not None,
                key=f"chk_{plat}"
            )
            if st.button(f"Save {plat}"):
                cur.execute(
                    "UPDATE prices SET manual_override=? WHERE phone_id=? AND platform=?",
                    (override if use_override else None, pid, plat),
                )
                con.commit()
                st.success(f"Saved pricing for {plat}")


def simulated_list(phone, platform: str, price: float) -> Tuple[str, str]:
    """Return (status, reason). Apply business rules for listing attempts."""
    # Stock & availability
    available = int(phone["stock"]) - int(phone["reserved_b2b"]) if phone else 0
    if phone["discontinued"]:
        return "failed", "Product discontinued"
    if available <= 0:
        return "failed", "Out of stock (B2B/direct reservation)"

    # Profitability check: must be positive profit
    profit = compute_profit(price, phone["cost_price"], platform)
    if profit <= 0:
        return "failed", "Unprofitable due to fees"

    # Condition support
    mapped = COND_MAP[platform].get(phone["condition"])
    if mapped is None:
        return "failed", "Unsupported condition"

    return "listed", "OK"


def page_platforms():
    st.header("🧪 Dummy Platform Integration")
    with get_conn() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM phones ORDER BY id DESC")
        phones = cur.fetchall()
        if not phones:
            st.info("Add phones first in Inventory page.")
            return

        # Search & filters
        st.subheader("🔎 Search & Filter Inventory")
        q = st.text_input("Search (brand/model)")
        cond = st.multiselect("Filter by condition", CONDITIONS)

        rows = []
        for r in phones:
            d = phone_to_series(r)
            if q:
                if q.lower() not in (d["brand"].lower() + " " + d["model"].lower()):
                    continue
            if cond and d["condition"] not in cond:
                continue
            rows.append(d)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Pick phone to list
        st.subheader("📤 Attempt Listing")
        options = [(d["id"], f"#{d['id']} {d['brand']} {d['model']} ({d['condition']})") for d in rows] if rows else []
        if not options:
            st.info("No phones match the filters.")
            return
        sel = st.selectbox("Select phone", options=options, format_func=lambda x: x[1])
        pid = sel[0]

   # Choose platform and compute price to use (override > auto)
plat = st.selectbox("Platform", options=PLATFORMS)

# Select phone
with get_conn() as con:
    cur = con.cursor()
    cur.execute("SELECT id, brand, model FROM phones")
    phones = cur.fetchall()

phone_choice = st.selectbox("Phone", options=phones, format_func=lambda x: f"{x[1]} {x[2]}")
if phone_choice:
    pid = phone_choice[0]

    with get_conn() as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM phones WHERE id=?", (pid,))
        phone = phone_to_series(cur.fetchone())

        cur.execute("SELECT auto_price, manual_override FROM prices WHERE phone_id=? AND platform=?", (pid, plat))
        pr = cur.fetchone()

    if not pr or pr[0] is None:
        auto = compute_listing_price(phone["base_net"], plat).price
    else:
        auto = pr[0]

    use_price = pr[1] if pr and pr[1] is not None else auto
    st.write(f"Using price: **{use_price}** (auto: {auto}, override: {pr[1] if pr else None})")

    if st.button("Simulate Listing"):
        status, reason = simulated_list(phone, plat, use_price)
        with get_conn() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO listings(phone_id, platform, status, reason) VALUES(?,?,?,?)",
                (pid, plat, status, reason),
            )
            con.commit()

        if status == "listed":
            st.success(f"Listed on {plat}. Condition mapped as: {COND_MAP[plat][phone['condition']]}")
        else:
            st.error(f"Failed to list on {plat}: {reason}")

    st.subheader("📊 Listing Audit")
    with get_conn() as con:
        cur = con.cursor()
        cur.execute(
            """
            SELECT l.phone_id, p.brand, p.model, l.platform, l.status, l.reason
            FROM listings l JOIN phones p ON l.phone_id=p.id
            ORDER BY l.rowid DESC
            """
        )
        audit = cur.fetchall()

    if audit:
        adf = pd.DataFrame([{
            "Phone#": a[0], "Brand": a[1], "Model": a[2],
            "Platform": a[3], "Status": a[4], "Reason": a[5]
        } for a in audit])
        st.dataframe(adf, use_container_width=True)
    else:
        st.info("No listing attempts yet.")




def page_condition_mapping():
    st.header("🧩 Condition Mapping")
    st.write("Internal conditions → platform categories")
    map_rows = []
    for plat in PLATFORMS:
        for c in CONDITIONS:
            map_rows.append({"Platform": plat, "Internal": c, "Platform Category": COND_MAP[plat].get(c, "Unsupported")})
    st.dataframe(pd.DataFrame(map_rows), use_container_width=True)

# ---------- App ----------


def main():
        st.set_page_config(page_title="Refurbished Phone Seller", layout="wide")
init_db()
seed_admin()


auth = login_form()
if not auth:
        st.stop()


        st.title("Refurbished Phone Selling – Dummy Integrator")
        st.caption("Manage inventory, compute platform prices, and simulate listings for platforms X, Y, Z.")


page = st.sidebar.radio("Navigate", ["Inventory", "Pricing", "Platforms", "Condition Mapping"])
if page == "Inventory":
        page_inventory()
elif page == "Pricing":
        page_pricing()
elif page == "Platforms":
        page_platforms()
else:
        page_condition_mapping()

if __name__ == "__main__":
 main()