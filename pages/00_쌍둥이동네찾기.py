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
COMPARE_COLORS = ["#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]


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


@st.cache_data
def compute_proportion_matrix(long_df: pd.DataFrame, ym: str, gender: str = "계"):
    """지역 x 연령(0~100) 인구 비율 행렬. 총인구수가 0인 지역은 제외."""
    sub = long_df[(long_df["기준년월"] == ym) & (long_df["성별"] == gender)]
    pivot = sub.pivot_table(index="행정구역", columns="연령", values="인구수", aggfunc="sum")
    pivot = pivot.reindex(columns=range(0, 101), fill_value=0)
    totals = pivot.sum(axis=1)
    valid = totals > 0
    prop = pivot.loc[valid].div(totals[valid], axis=0)
    return prop, totals[valid]


def find_similar_regions(prop_df: pd.DataFrame, target: str, top_n: int, use_cosine: bool) -> pd.DataFrame:
    target_vec = prop_df.loc[target].values
    others = prop_df.drop(index=target)

    if use_cosine:
        norms = np.linalg.norm(others.values, axis=1) * np.linalg.norm(target_vec)
        norms[norms == 0] = np.nan
        score = (others.values @ target_vec) / norms
        score_col, ascending = "유사도(코사인)", False
    else:
        score = np.linalg.norm(others.values - target_vec, axis=1)
        score_col, ascending = "구조 차이(유클리드 거리)", True

    result = pd.DataFrame({"행정구역": others.index, score_col: score})
    result = result.sort_values(score_col, ascending=ascending).head(top_n).reset_index(drop=True)
    result.insert(0, "순위", range(1, len(result) + 1))
    return result, score_col


def region_search_select(label_prefix: str, all_regions: list, default: str, key: str) -> str:
    keyword = st.text_input(f"{label_prefix} 검색 (입력)", placeholder="예: 종로, 해운대, 수원", key=f"{key}_kw")
    options = [r for r in all_regions if keyword.strip() in r] if keyword.strip() else all_regions
    if not options:
        options = all_regions
    index = options.index(default) if default in options else 0
    return st.selectbox(f"{label_prefix} 선택", options=options, index=index, key=f"{key}_sel")


def main():
    st.title("지역별 인구구조 대시보드")
    st.caption(
        "행정안전부 '연령별 인구현황(월간)' 데이터를 바탕으로 지역별 연령 구조를 비교하고, "
        "전국에서 인구구조가 가장 비슷한 지역 Top N을 찾아볼 수 있습니다."
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

    all_regions = df["행정구역"].tolist()

    with st.sidebar:
        st.header("공통 조회 조건")
        ym = st.selectbox("기준년월", year_months, index=len(year_months) - 1)

    # ------------------------------------------------------------------
    # 1. 지역별 인구구조 비교 (꺾은선 그래프)
    # ------------------------------------------------------------------
    st.header("1. 지역별 인구구조 비교")

    with st.sidebar:
        st.subheader("① 지역 비교 옵션")
        genders = st.multiselect(
            "성별", options=["계", "남", "여"], default=["계"], format_func=lambda g: GENDER_LABEL[g], key="cmp_gender"
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        keyword = st.text_input("지역명 검색 (입력)", placeholder="예: 종로, 해운대, 수원", key="cmp_kw")
    filtered_regions = [r for r in all_regions if keyword.strip() in r] if keyword.strip() else all_regions

    with col2:
        default_region = [filtered_regions[0]] if filtered_regions else []
        selected_regions = st.multiselect(
            "지역 선택 (검색 결과 중 선택, 여러 지역 비교 가능)",
            options=filtered_regions,
            default=default_region,
            key="cmp_regions",
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

    fig1 = go.Figure()
    dash_options = ["solid", "dash", "dot", "dashdot", "longdash"]
    for i, region in enumerate(selected_regions):
        dash = dash_options[i % len(dash_options)]
        for gender in genders:
            sub = plot_df[(plot_df["행정구역"] == region) & (plot_df["성별"] == gender)]
            if sub.empty:
                continue
            fig1.add_trace(
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

    fig1.update_layout(
        title=f"{ym} 연령별 인구구조",
        xaxis_title="연령 (세, 100=100세 이상)",
        yaxis_title="인구수 (명)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("데이터 테이블 보기"):
        pivot = plot_df.pivot_table(index="연령", columns=["행정구역", "성별"], values="인구수")
        st.dataframe(pivot, use_container_width=True)

    # ------------------------------------------------------------------
    # 2. 전국 유사 인구구조 Top N
    # ------------------------------------------------------------------
    st.header("2. 전국에서 인구구조가 가장 비슷한 지역")

    with st.sidebar:
        st.subheader("② 유사 지역 찾기 옵션")
        top_n = st.slider("Top N", min_value=3, max_value=10, value=5)
        use_cosine = st.radio(
            "유사도 기준",
            options=[False, True],
            format_func=lambda v: "코사인 유사도 (형태 유사)" if v else "유클리드 거리 (구조 차이, 추천)",
        )

    default_target = selected_regions[0] if selected_regions else all_regions[0]
    target_region = region_search_select("기준 지역", all_regions, default_target, key="target")

    prop_df, totals = compute_proportion_matrix(long_df, ym, gender="계")

    if target_region not in prop_df.index:
        st.warning(f"'{target_region}'은(는) 총인구수가 0이라 구조 비교가 불가능합니다.")
    else:
        result, score_col = find_similar_regions(prop_df, target_region, top_n, use_cosine)
        result["총인구수"] = result["행정구역"].map(totals).map(lambda v: f"{v:,.0f}")

        col_table, col_chart = st.columns([1, 2])
        with col_table:
            st.markdown(f"**'{target_region}'** 와(과) 인구구조가 가장 비슷한 지역 Top {top_n}")
            st.dataframe(result.set_index("순위"), use_container_width=True)

        with col_chart:
            fig2 = go.Figure(
                go.Bar(
                    x=result[score_col],
                    y=[f"{row['순위']}. {row['행정구역']}" for _, row in result.iterrows()],
                    orientation="h",
                    marker_color=COMPARE_COLORS[: len(result)],
                )
            )
            fig2.update_layout(
                title=f"'{target_region}' 유사도 순위 ({score_col})",
                xaxis_title=score_col,
                yaxis=dict(autorange="reversed"),
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 연령 구조(비율) 오버레이 비교
        fig3 = go.Figure()
        ages = list(range(0, 101))
        fig3.add_trace(
            go.Scatter(
                x=ages,
                y=prop_df.loc[target_region].values * 100,
                mode="lines",
                name=f"[기준] {target_region}",
                line=dict(color="#111111", width=3),
                hovertemplate="연령 %{x}세<br>비율 %{y:.2f}%<extra>" + target_region + "</extra>",
            )
        )
        for i, row in result.iterrows():
            region = row["행정구역"]
            fig3.add_trace(
                go.Scatter(
                    x=ages,
                    y=prop_df.loc[region].values * 100,
                    mode="lines",
                    name=f"{row['순위']}위. {region}",
                    line=dict(color=COMPARE_COLORS[i % len(COMPARE_COLORS)], dash="dot"),
                    hovertemplate="연령 %{x}세<br>비율 %{y:.2f}%<extra>" + region + "</extra>",
                )
            )
        fig3.update_layout(
            title=f"{ym} 연령대별 인구 비율(%) 비교 — 기준 지역 vs Top {top_n}",
            xaxis_title="연령 (세, 100=100세 이상)",
            yaxis_title="해당 지역 내 인구 비율 (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "인구구조 유사도는 각 지역의 연령별 인구 '비율'(0~100세, 100세는 100세 이상 합산)을 기준으로 계산합니다. "
            "'유클리드 거리'는 값이 작을수록, '코사인 유사도'는 값이 1에 가까울수록 구조가 비슷함을 의미합니다."
        )


if __name__ == "__main__":
    main()
