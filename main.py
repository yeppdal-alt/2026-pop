import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="지역별 인구구조 대시보드", layout="wide")

DATA_FILE = Path(__file__).parent / "202606_202606_연령별인구현황_월간.csv"
ENCODING_CANDIDATES = ["cp949", "euc-kr", "utf-8"]

GENDER_LABEL = {"계": "전체", "남": "남자", "여": "여자"}
GENDER_COLOR = {"계": "#636EFA", "남": "#00B5F7", "여": "#EF553B"}
AGE_PATTERN = re.compile(r"^(\d{4}년\d{2}월)_(계|남|여)_(\d+)세(?:\s*이상)?$")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def clean_number(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"-": np.nan, "": np.nan})
        .astype(float)
    )


@st.cache_data
def parse_long(df: pd.DataFrame):
    """행정안전부 연령별 인구현황 원본(wide)을 (지역, 기준월, 성별, 연령, 인구수) long 포맷으로 변환."""
    matches = [(col, *AGE_PATTERN.match(col).groups()) for col in df.columns if AGE_PATTERN.match(col)]
    if not matches:
        return pd.DataFrame(), []

    frames = []
    for col, ym, gender, age in matches:
        frames.append(
            pd.DataFrame(
                {
                    "행정구역": df["행정구역"],
                    "기준년월": ym,
                    "성별": gender,
                    "연령": int(age),
                    "인구수": clean_number(df[col]),
                }
            )
        )
    long_df = pd.concat(frames, ignore_index=True)
    year_months = sorted(long_df["기준년월"].unique())
    return long_df, year_months


def main():
    st.title("지역별 인구구조 대시보드")
    st.caption(
        "행정안전부 '연령별 인구현황(월간)' 데이터를 바탕으로, 원하는 지역을 검색·선택해 "
        "연령별 인구구조를 꺾은선 그래프로 볼 수 있습니다."
    )

    if not DATA_FILE.exists():
        st.error(
            f"데이터 파일을 찾을 수 없습니다: '{DATA_FILE.name}'\n\n"
            "이 CSV 파일을 app.py와 같은 폴더에 넣어주세요."
        )
        st.stop()

    try:
        df = load_data(DATA_FILE)
    except UnicodeDecodeError:
        st.error("CSV 인코딩을 인식하지 못했습니다 (cp949 / euc-kr / utf-8 모두 실패).")
        st.stop()

    if "행정구역" not in df.columns:
        st.error("'행정구역' 컬럼을 찾을 수 없습니다. 올바른 형식의 CSV인지 확인해주세요.")
        st.stop()

    long_df, year_months = parse_long(df)
    if long_df.empty:
        st.error("연령별 인구수 컬럼을 찾지 못했습니다. CSV 형식을 확인해주세요.")
        st.stop()

    with st.sidebar:
        st.header("조회 조건")
        ym = st.selectbox("기준년월", year_months, index=len(year_months) - 1)
        genders = st.multiselect(
            "성별", options=["계", "남", "여"], default=["계"], format_func=lambda g: GENDER_LABEL[g]
        )

    st.subheader("지역 선택")
    all_regions = df["행정구역"].tolist()

    col1, col2 = st.columns([1, 2])
    with col1:
        keyword = st.text_input("지역명 검색 (입력)", placeholder="예: 종로, 해운대, 수원")
    filtered_regions = [r for r in all_regions if keyword.strip() in r] if keyword.strip() else all_regions

    with col2:
        default_region = [filtered_regions[0]] if filtered_regions else []
        selected_regions = st.multiselect(
            "지역 선택 (검색 결과 중 선택, 여러 지역 비교 가능)",
            options=filtered_regions,
            default=default_region,
        )

    if not selected_regions:
        st.warning("지역을 최소 1개 이상 선택해주세요.")
        st.stop()
    if not genders:
        st.warning("성별을 최소 1개 이상 선택해주세요.")
        st.stop()

    plot_df = long_df[
        (long_df["기준년월"] == ym)
        & (long_df["행정구역"].isin(selected_regions))
        & (long_df["성별"].isin(genders))
    ].sort_values("연령")

    fig = go.Figure()
    dash_options = ["solid", "dash", "dot", "dashdot", "longdash"]
    for i, region in enumerate(selected_regions):
        dash = dash_options[i % len(dash_options)]
        for gender in genders:
            sub = plot_df[(plot_df["행정구역"] == region) & (plot_df["성별"] == gender)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["연령"],
                    y=sub["인구수"],
                    mode="lines",
                    name=f"{region} - {GENDER_LABEL[gender]}",
                    line=dict(color=GENDER_COLOR[gender], dash=dash),
                    hovertemplate="연령 %{x}세<br>인구수 %{y:,.0f}명<extra>"
                    + f"{region} - {GENDER_LABEL[gender]}"
                    + "</extra>",
                )
            )

    fig.update_layout(
        title=f"{ym} 연령별 인구구조",
        xaxis_title="연령 (세, 100=100세 이상)",
        yaxis_title="인구수 (명)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("데이터 테이블 보기"):
        pivot = plot_df.pivot_table(index="연령", columns=["행정구역", "성별"], values="인구수")
        st.dataframe(pivot, use_container_width=True)


if __name__ == "__main__":
    main()
