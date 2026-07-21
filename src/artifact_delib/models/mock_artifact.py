"""Deterministic mock for artifact identification — no real API calls.

Uses a call counter to simulate progressive improvement across pipeline rounds.
"""

from __future__ import annotations

from artifact_delib.api.schemas import ModelRequest, ModelResponse, TokenUsage

# ── Round 1 (initial) content ──
_R1 = {
    "visual_perception": (
        "图片中呈现一件完整的陶瓷器物，整体器形呈瓶状。"
        "器物表面以蓝白配色为主，可见明显的手绘纹饰装饰。"
        "器表光泽自然，底部有圈足结构。"
    ),
    "shape_expert": (
        "该器物整体轮廓清晰，具有典型的小口、短颈、丰肩特征，"
        "腹部向下逐渐收束，底部设有圈足。整体比例协调，肩部线条饱满圆润，"
        "符合传统梅瓶的基本器形特征。口沿部分略向外撇，颈部短而直，"
        "圈足较矮且外壁垂直。"
    ),
    "style_expert": (
        "器物表面以釉下青花进行装饰，主体纹饰为缠枝莲图案，"
        "构图采用分层布局，腹部为缠枝莲主题纹样，肩部和胫部辅以莲瓣纹。"
        "青花发色较为浓艳，具有典型的苏麻离青特征。纹饰线条流畅，"
        "布局疏密有致，整体呈现明代早期青花瓷的典型风格特征。"
    ),
    "glyph_expert": (
        "器底可见一处方形双框款识区域，框内疑似存在六字竖向排列的款文。"
        "当前图像分辨率和角度限制下，无法完全辨认具体文字内容。"
        "款识区域存在明显釉面覆盖特征，文字周边有轻微积釉现象。"
        "该区域位置和布局与明代官窑款识特征基本吻合，但具体年代无法仅凭款识位置确认。"
    ),
    "material_expert": (
        "从视觉特征判断，该器物应为瓷质，表面施以透明釉，"
        "釉层均匀且光泽度良好。釉下可见青花着色区域，呈色较为稳定。"
        "胎体从口沿露胎处观察呈白色，质地较为细腻。"
        "器物底部露胎处可见细密旋坯痕迹。器表未见明显开片或剥釉现象。"
    ),
    "local_detail_expert": (
        "器底圈足内墙可见细小旋纹，足端露胎处呈现浅火石红色，"
        "此为部分明代瓷器的常见特征。腹部缠枝莲纹样中，莲瓣尖端有轻微积釉现象。"
        "器物内壁可见拉坯痕迹，整体器壁厚度均匀。"
        "口沿处釉面有轻微垂釉现象。器表未见明显磨损或修复痕迹。"
    ),
    "summarizer": (
        "综合各视觉专家观察，该器物整体具有典型梅瓶轮廓，小口、短颈、丰肩并向下逐渐收束。"
        "器表采用蓝色釉下装饰，主体纹样表现出缠枝莲特征，整体视觉风格较接近明代早期青花瓷。"
        "器物底部疑似存在方形款识区域，当前图像不足以辨认具体文字。"
        "器形和纹饰均支持其属于明初梅瓶，但尚不足以稳定区分永乐和宣德时期。"
    ),
    "candidate_generator": (
        "基于综合视觉分析，当前最可能的候选是明永乐时期青花梅瓶。"
        "第二个候选是明宣德时期青花梅瓶。"
        "第三个候选是更宽泛的明代早期青花梅瓶。"
        "\n\n```json\n"
        '{"candidates": ['
        '{"text": "明永乐青花梅瓶", "confidence": 0.48},'
        '{"text": "明宣德青花梅瓶", "confidence": 0.32},'
        '{"text": "明代早期青花梅瓶", "confidence": 0.20}'
        "]}\n```"
    ),
    "disagreement_analyzer": (
        "当前两个主要候选都认为该器物属于明代早期青花梅瓶，器型本身不存在明显分歧。"
        "主要不确定性集中在永乐和宣德的具体年代判断。"
        "现有纹饰特征尚不足以完成稳定区分，因此下一步最值得重新分析纹饰布局和整体青花风格。"
        "\n分歧类型：STYLE"
    ),
    "judge": (
        "综合器形、纹饰和整体视觉风格来看，该器物最可能是一件明代早期青花梅瓶，"
        "具体年代更倾向于永乐时期。其小口、短颈、丰肩和下腹逐渐收束的器形符合梅瓶特征，"
        "缠枝莲纹的布局及青花表现也与明初作品较为接近。"
        "相比宣德候选，现有视觉信息略微支持永乐，但由于缺少清晰款识和更多器底细节，"
        "具体年代仍存在一定不确定性。"
    ),
    "hypothesis_agent": (
        "基于现有专家分析，我支持当前的候选判断。"
        "器形、纹饰和局部细节都指向这一方向。"
        "\n立场：MAINTAIN"
    ),
    "critic": (
        "双方均基于已有专家报告，没有新增有效区分信息。"
        "\n继续：否"
    ),
    "direct_vlm": (
        "该器物整体呈瓶状，蓝白配色，底部有圈足。"
        "综合判断为明代早期青花梅瓶，可能为永乐时期。"
    ),
    "generic_mad": (
        "基于视觉特征分析，该器物为一件青花瓷梅瓶。"
        "纹饰风格倾向于明代早期，与永乐时期特征较为吻合。"
    ),
}

# ── Round 2 (after first recheck) — targeted expert gets a focused recheck response ──
# Other modules see slightly improved aggregate results
_R2 = {
    "shape_expert": (
        "再次仔细查看器形细节。两个候选均属于梅瓶器形，在整体轮廓上高度一致。"
        "永乐梅瓶肩部通常较宣德更为饱满圆润，腹部曲线过渡更为自然。"
        "当前器物的肩部线条确实较为丰满，腹部收束曲线流畅，"
        "略微倾向于永乐时期的器形特征。但仅凭器形曲线仍不足以做出决定性判断。"
    ),
    "style_expert": (
        "重点比较纹饰细节以区分永乐和宣德时期。"
        "永乐时期缠枝莲纹的莲瓣通常较为饱满，叶片肥厚，整体构图疏朗；"
        "宣德时期纹饰趋于繁密，青花发色略深，笔触更为刚劲。"
        "当前器物纹饰布局较为疏朗，莲瓣形态饱满，青花发色浓艳但不过深，"
        "整体风格更接近永乐时期的典型特征。"
    ),
    "glyph_expert": (
        "重新审视款识区域。虽然无法完全辨认具体文字，"
        "但注意到款识边框的边角处理和釉面覆盖特征。"
        "永乐器底款识通常为暗刻或淡描，宣德款识则较为清晰有力。"
        "当前器物款识区域的釉面覆盖程度和线条力度更接近永乐器物的特征。"
    ),
    "material_expert": (
        "从材质角度比较两个候选。永乐和宣德青花均使用进口苏麻离青料，"
        "但在烧制气氛上存在差异。永乐时期釉面更显肥润，呈色略偏蓝绿；"
        "宣德时期釉面稍显青白，积釉处呈水绿色。"
        "当前器物釉面肥润感较强，青花发色蓝中略泛紫，更接近于永乐时期的釉面特征。"
    ),
    "local_detail_expert": (
        "重点查看器底和足部细节。永乐时期圈足内墙通常呈斜坡状，"
        "足端露胎处呈现浅淡的橙红色；宣德时期圈足更为垂直，露胎处颜色偏深。"
        "当前器物圈足内墙呈明显斜坡状，足端露胎呈浅火石红色，"
        "此特征倾向于永乐时期的工艺特点。"
    ),
    "summarizer": (
        "经定向重审后，各专家对区分特征进行了更细致比较。"
        "纹饰布局疏朗、莲瓣饱满、青花发色浓艳但不深，倾向永乐时期特征。"
        "器形方面肩部丰满、腹部曲线流畅，也略微倾向永乐。"
        "综合来看，多个维度的视觉信息均趋向于永乐时期，但仍有少许不确定性。"
    ),
    "candidate_generator": (
        "经重新分析，纹饰特征和局部细节进一步支持明永乐青花梅瓶的判断。"
        "第二个候选仍为明宣德青花梅瓶，但置信度差距有所扩大。"
        "\n\n```json\n"
        '{"candidates": ['
        '{"text": "明永乐青花梅瓶", "confidence": 0.58},'
        '{"text": "明宣德青花梅瓶", "confidence": 0.28},'
        '{"text": "明代早期青花梅瓶", "confidence": 0.14}'
    "]}\n```"),
    "disagreement_analyzer": (
        "经定向重审后，纹饰风格和局部细节分析进一步支持永乐时期的判断。"
        "当前两个候选仍然存在但置信度差距有所扩大。"
        "主要不确定性已从多方面分歧缩小为年代细节的确认。"
        "\n分歧类型：STYLE"
    ),
    "judge": (
        "经过定向重审后，综合多维度分析结果更为明确。"
        "该器物应鉴定为一件明代永乐时期的青花梅瓶。"
        "器形方面肩部饱满、腹部曲线流畅，符合永乐梅瓶的典型特征。"
        "纹饰方面缠枝莲布局疏朗、莲瓣饱满，青花发色浓艳而清亮，"
        "与永乐时期青花瓷的风格高度吻合。"
        "虽然缺少款识的确凿证据，但器形、纹饰、釉面和圈足细节相互印证，"
        "共同指向永乐时期的判断。"
    ),
    "hypothesis_agent": (
        "经过定向重审后，多个维度的证据进一步强化了当前判断。"
        "器形、纹饰、局部细节相互印证，形成一致指向。"
        "\n立场：MAINTAIN"
    ),
    "critic": (
        "双方均表达了一致立场，没有实质分歧需要进一步讨论。"
        "\n继续：否"
    ),
}


class ArtifactMockClient:
    """Return deterministic artifact analysis content.

    Uses a call counter to simulate improving confidence across recheck rounds.
    Calls 1-9  → Round 1 content (low confidence, high uncertainty)
    Calls 10+  → Round 2 content (improved confidence, narrower uncertainty)
    """

    def __init__(self, role: str = "artifact_vlm") -> None:
        self.role = role
        self._call_count = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._call_count += 1
        content = self._select_content(request)
        usage = TokenUsage(
            input_tokens=max(1, len(str(request.user_prompt).split())),
            output_tokens=max(1, len(content.split())),
        )
        return ModelResponse(
            request_id=request.request_id,
            content=content,
            usage=usage,
            latency_ms=0.0,
            provider="mock",
            model=request.model,
        )

    def _select_content(self, request: ModelRequest) -> str:
        prompt_name = (request.prompt_name or "").lower()
        is_recheck = self._call_count > 9  # After initial 9 calls, use R2

        source = _R2 if is_recheck else _R1

        # Exact key match
        if prompt_name in source:
            return source[prompt_name]

        # Partial key match
        for key in source:
            if key in prompt_name or prompt_name in key:
                return source[key]

        # Text fallback
        text = (request.system_prompt + " " + request.user_prompt).lower()
        for key, content in source.items():
            kw = key.replace("_expert", "").replace("_", "")
            if kw in text:
                return content
        return source.get("visual_perception", list(source.values())[0] if source else "观察描述。")
