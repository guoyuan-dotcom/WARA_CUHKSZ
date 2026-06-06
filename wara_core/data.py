from __future__ import annotations

from typing import Any


_WIRELESS_SEMINAL_PAPERS: tuple[dict[str, Any], ...] = (
    {
        "title": "MIMO Broadcasting for Simultaneous Wireless Information and Power Transfer",
        "authors": "R. Zhang and C. K. Ho",
        "year": 2013,
        "venue": "IEEE Transactions on Wireless Communications",
        "cite_key": "ZhangHo2013",
        "keywords": ["swipt", "wireless information and power transfer", "beamforming", "power splitting"],
    },
    {
        "title": "Wireless Information and Power Transfer: Architecture Design and Rate-Energy Tradeoff",
        "authors": "X. Zhou, R. Zhang, and C. K. Ho",
        "year": 2013,
        "venue": "IEEE Transactions on Communications",
        "cite_key": "ZhouZhangHo2013",
        "keywords": ["swipt", "rate-energy", "power splitting", "energy harvesting"],
    },
    {
        "title": "Joint Transmit Beamforming and Receive Power Splitting for MISO SWIPT Systems",
        "authors": "Q. Shi, L. Liu, W. Xu, and R. Zhang",
        "year": 2014,
        "venue": "IEEE Transactions on Wireless Communications",
        "cite_key": "ShiLiuXuZhang2014",
        "keywords": ["swipt", "beamforming", "power splitting", "miso"],
    },
    {
        "title": "Practical Non-Linear Energy Harvesting Model and Resource Allocation for SWIPT Systems",
        "authors": "E. Boshkovska, D. W. K. Ng, N. Zlatanov, and R. Schober",
        "year": 2015,
        "venue": "IEEE Communications Letters",
        "cite_key": "Boshkovska2015",
        "keywords": ["swipt", "nonlinear energy harvesting", "non-linear energy harvesting", "rectifier"],
    },
    {
        "title": "Fundamentals of Wireless Information and Power Transfer: From RF Energy Harvester Models to Signal and System Designs",
        "authors": "B. Clerckx, R. Zhang, R. Schober, D. W. K. Ng, D. I. Kim, and H. V. Poor",
        "year": 2019,
        "venue": "IEEE Journal on Selected Areas in Communications",
        "cite_key": "Clerckx2019WIPT",
        "keywords": ["wireless power transfer", "wpt", "swipt", "energy harvesting", "rf energy harvester"],
    },
    {
        "title": "A Survey on Integrated Sensing and Communication",
        "authors": "F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi",
        "year": 2022,
        "venue": "IEEE Communications Surveys & Tutorials",
        "cite_key": "LiuCuiMasouros2022ISAC",
        "keywords": ["isac", "integrated sensing and communication", "sensing", "communication"],
    },
    {
        "title": "Joint Beamforming Design for Integrated Sensing and Communication Systems",
        "authors": "F. Liu, C. Masouros, A. P. Petropulu, H. Griffiths, and L. Hanzo",
        "year": 2020,
        "venue": "IEEE Transactions on Wireless Communications",
        "cite_key": "LiuMasourosPetropulu2020ISAC",
        "keywords": ["isac", "beamforming", "sensing", "communication"],
    },
    {
        "title": "Reconfigurable Intelligent Surfaces: Principles and Opportunities",
        "authors": "Q. Wu and R. Zhang",
        "year": 2020,
        "venue": "IEEE Communications Surveys & Tutorials",
        "cite_key": "WuZhang2020RIS",
        "keywords": ["ris", "irs", "reconfigurable intelligent surface", "intelligent reflecting surface"],
    },
    {
        "title": "Wireless Communications With Reconfigurable Intelligent Surface: Path Loss Modeling and Experimental Measurement",
        "authors": "W. Tang et al.",
        "year": 2021,
        "venue": "IEEE Transactions on Wireless Communications",
        "cite_key": "Tang2021RIS",
        "keywords": ["ris", "channel", "reconfigurable intelligent surface"],
    },
    {
        "title": "Energy-Efficient UAV Communication With Trajectory Optimization",
        "authors": "Y. Zeng and R. Zhang",
        "year": 2017,
        "venue": "IEEE Transactions on Wireless Communications",
        "cite_key": "ZengZhang2017UAV",
        "keywords": ["uav", "trajectory", "energy efficiency", "wireless communication"],
    },
    {
        "title": "A Survey on UAV Communications for 5G and Beyond",
        "authors": "Y. Zeng, R. Zhang, and T. J. Lim",
        "year": 2016,
        "venue": "IEEE Communications Surveys & Tutorials",
        "cite_key": "ZengZhangLim2016UAV",
        "keywords": ["uav", "wireless communication", "trajectory", "air-ground channel"],
    },
    {
        "title": "An Iteratively Weighted MMSE Approach to Distributed Sum-Utility Maximization for a MIMO Interfering Broadcast Channel",
        "authors": "Q. Shi, M. Razaviyayn, Z.-Q. Luo, and C. He",
        "year": 2011,
        "venue": "IEEE Transactions on Signal Processing",
        "cite_key": "ShiRazaviyaynLuoHe2011",
        "keywords": ["wmmse", "sum utility", "mimo", "beamforming", "nonconvex optimization"],
    },
)


def load_seminal_papers(topic: str) -> list[dict[str, Any]]:
    """Return WARA-native wireless seed references whose keywords match topic text."""

    topic_lower = str(topic or "").lower()
    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in _WIRELESS_SEMINAL_PAPERS:
        keywords = paper.get("keywords", [])
        if not isinstance(keywords, list):
            continue
        if any(str(keyword).lower() in topic_lower for keyword in keywords):
            key = str(paper.get("cite_key") or "")
            if key not in seen:
                seen.add(key)
                matched.append(dict(paper))
    return matched


__all__ = ["load_seminal_papers"]
