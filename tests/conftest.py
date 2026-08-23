"""pytest 共享 fixture 与测试数据生成。

PDF 相关测试需要一份 200+ 页、带章节层级的中文 PDF（模拟企业年报）。
由于不引入真实年报二进制，这里用 PyMuPDF 程序化生成一份
`tests/data/xiaomi.pdf`（首次请求时生成，之后复用）。

章节结构（供 Structure-aware 切分与真实评测集使用）：
    5 章 × 3 节 × 15 页 = 210 页（> 200）
每个节配相关正文，使真实 BGE 检索可按语义命中期望章节。
"""

from pathlib import Path

import pytest

_PDF_PATH = Path(__file__).parent / "data" / "xiaomi.pdf"
_PAGES_PER_SECTION = 15  # 5章×3节×15页 = 210 页

_CHAPTERS = [
    ("第一章 公司概况", ["1.1 公司简介", "1.2 主营业务范围", "1.3 公司治理"]),
    ("第二章 经营情况讨论与分析", ["2.1 经营成果", "2.2 财务状况", "2.3 现金流"]),
    ("第三章 未来战略规划", ["3.1 发展战略", "3.2 智能电动车业务升级目标"]),
    ("第四章 业务回顾", ["4.1 智能手机业务", "4.2 IoT业务", "4.3 互联网服务"]),
    ("第五章 财务报告", ["5.1 主要会计数据", "5.2 财务指标", "5.3 风险提示"]),
]

# 各节相关正文（供真实 BGE 检索命中）
_SECTION_BODY = {
    "1.1 公司简介": [
        "小米集团是一家以智能手机、IoT与生活消费产品为核心的全球化科技公司。",
        "公司成立于2010年，总部位于北京，坚持“硬件+新零售+互联网服务”模式。",
    ],
    "1.2 主营业务范围": [
        "主营业务涵盖智能手机、IoT与生活消费产品、互联网服务三大板块。",
        "智能手机是收入主力，全球市场份额持续提升；IoT生态链覆盖智能家居等多个品类。",
    ],
    "1.3 公司治理": [
        "公司治理结构完善，董事会下设审计委员会与薪酬委员会。",
        "独立非执行董事占比合理，持续加强内部控制与合规管理。",
    ],
    "2.1 经营成果": [
        "本年度营业收入实现稳健增长，创历史新高，净利润同比大幅提升。",
        "毛利率稳中有升，主要得益于产品结构优化与规模效应。",
    ],
    "2.2 财务状况": [
        "公司资产负债结构稳健，有息负债占比低，经营性现金流充裕。",
        "应收账款与存货周转效率持续优化，为研发投入提供保障。",
    ],
    "2.3 现金流": [
        "经营活动产生的现金流量净额显著增长。",
        "投资活动聚焦产能扩张与核心技术研发，筹资活动保持审慎。",
    ],
    "3.1 发展战略": [
        "公司坚持高端化战略，持续提升品牌力与技术自研能力。",
        "重点投入人工智能、智能制造等前沿领域，全球化布局稳步推进。",
    ],
    "3.2 智能电动车业务升级目标": [
        "小米智能电动车业务加速推进，目标成为行业头部玩家。",
        "智能驾驶与智能座舱技术持续升级，首款车型已上市交付并快速扩张。",
    ],
    "4.1 智能手机业务": [
        "小米智能手机全球出货量保持增长，高端机型占比与平均售价持续提升。",
        "海外市场表现强劲，多个地区份额位居行业前列。",
    ],
    "4.2 IoT业务": [
        "IoT与生活消费产品业务收入持续增长，生态链不断丰富。",
        "智能家居、智能可穿戴设备出货量领先，平台连接设备数创新高。",
    ],
    "4.3 互联网服务": [
        "互联网服务收入稳步增长，广告与游戏业务表现亮眼。",
        "用户规模持续扩大，月活跃用户数创新高，会员渗透率提升。",
    ],
    "5.1 主要会计数据": [
        "本年度营业收入同比增长显著，净利润大幅增长，总资产规模扩大。",
        "研发投入占比处于行业领先水平，净资产收益率保持稳健。",
    ],
    "5.2 财务指标": [
        "毛利率、净利率、净资产收益率等核心指标稳健。",
        "资产负债率保持合理水平，流动性充裕，每股收益稳步提升。",
    ],
    "5.3 风险提示": [
        "公司面临的经营风险包括行业竞争加剧与宏观环境变化。",
        "原材料价格波动可能影响成本与毛利率，海外政策存在不确定性。",
    ],
}


def _total_pages() -> int:
    return sum(len(secs) for _, secs in _CHAPTERS) * _PAGES_PER_SECTION


def _structure_version() -> str:
    """结构配置的版本指纹：章节/正文/页数任一变化 → 重新生成 PDF。"""
    import hashlib
    import json

    raw = json.dumps(
        {"chapters": _CHAPTERS, "body": _SECTION_BODY, "pps": _PAGES_PER_SECTION},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode()).hexdigest()


_VERSION_FILE = _PDF_PATH.with_name(_PDF_PATH.name + ".version")


def _ensure_xiaomi_pdf() -> Path:
    """生成带章节结构的多页中文模拟年报 PDF；版本未变时复用缓存。"""
    if (
        _PDF_PATH.exists()
        and _VERSION_FILE.exists()
        and _VERSION_FILE.read_text(encoding="utf-8") == _structure_version()
    ):
        return _PDF_PATH

    import fitz  # PyMuPDF

    _PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        page_no = 1
        for ch_title, sec_titles in _CHAPTERS:
            for sec_title in sec_titles:
                body = _SECTION_BODY[sec_title]
                for pi in range(_PAGES_PER_SECTION):
                    page = doc.new_page()
                    y = 72
                    page.insert_text(
                        (72, y), f"小米集团年度报告（模拟） 第{page_no}页",
                        fontname="china-s", fontsize=14,
                    )
                    y += 40
                    if pi == 0:
                        page.insert_text((72, y), ch_title, fontname="china-s", fontsize=12)
                        y += 30
                        page.insert_text((72, y), sec_title, fontname="china-s", fontsize=12)
                        y += 30
                    for line in body:
                        page.insert_text((72, y), line, fontname="china-s", fontsize=10)
                        y += 22
                    page_no += 1
        doc.save(str(_PDF_PATH))
        _VERSION_FILE.write_text(_structure_version(), encoding="utf-8")
    finally:
        doc.close()

    return _PDF_PATH


@pytest.fixture(scope="session")
def xiaomi_pdf_path():
    """会话级 fixture：带章节结构的多页中文模拟年报 PDF 路径。"""
    try:
        return _ensure_xiaomi_pdf()
    except ImportError:
        pytest.skip("pymupdf 未安装，跳过 PDF 相关测试")


# ── 测试默认精排器：DummyReranker（避免加载 2.2GB CrossEncoder）─────────
# .env 可能设置了真实模型路径；非 real 测试一律强制 dummy。
# real 测试（如 CATL 评测）在用例内显式启用真实模型。


@pytest.fixture(autouse=True)
def _force_dummy_reranker_for_tests(monkeypatch):
    import app.rag.reranker as rr
    from app.core.config import settings

    monkeypatch.setattr(settings, "rag_reranker_model", "dummy")
    monkeypatch.setattr(rr, "_default_reranker", None)


# ── 真实模型测试开关（real marker）─────────────────────────────
# 需要真实 BGE-M3 的慢测试默认跳过，用 `pytest --run-real` 运行。


def pytest_addoption(parser):
    parser.addoption(
        "--run-real",
        action="store_true",
        default=False,
        help="运行需要真实模型（BGE-M3）的慢测试",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-real"):
        skip_real = pytest.mark.skip(reason="需要真实模型，用 --run-real 运行")
        for item in items:
            if "real" in item.keywords:
                item.add_marker(skip_real)
