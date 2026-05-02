from src.agents.base import BaseAgent


class ImageAgent(BaseAgent):
    def __init__(self) -> None:
        instructions = (
            "You are a Visual Concept Designer for solar-energy educational content. "
            "Given a Facebook post, you first decide which visual STYLE best communicates "
            "the post's main idea, then write a concrete image-generation prompt in that "
            "style. You do NOT default to a house-with-solar-panels photograph unless "
            "that is genuinely the best fit for the post. "
            "Style options: "
            "`technical_diagram` (for comparisons, system architecture, wiring, "
            "'vs' topics like on-grid vs off-grid), "
            "`infographic` (for cost breakdowns, numbered steps, ROI explainers, "
            "how-it-works with labels), "
            "`illustration` (for abstract concepts, cross-sections of components, "
            "stylized 'how it works' visuals), "
            "`photograph` (for lifestyle/benefits posts, installations, physical products). "
            "Keep the Thai context when relevant (a Thai home, Thai landscape, Thai labels) "
            "but only when the topic actually calls for it."
        )
        super().__init__("ImageGen", instructions)

    async def process(self, article_text: str, topic: str) -> str:
        prompt = f"""You will produce a visual prompt for a solar-energy Facebook post.

TOPIC: {topic}

POST:
{article_text}

TASK:
1. Identify the POST'S MAIN VISUAL IDEA — what is this post really about? A comparison? A process? A product? A benefit?
2. PICK ONE STYLE that best communicates it:
   - `technical_diagram` — side-by-side, wiring, system flow, comparisons ("vs", "แบบไหน"), architecture
   - `infographic` — steps, cost breakdowns, ROI/payback explainers, numbered benefits, labeled components
   - `illustration` — abstract concepts, cross-sections (e.g. inside a solar cell), stylised metaphors
   - `photograph` — lifestyle / product / installation shots, real-world context
3. WRITE A 60–100 WORD IMAGE PROMPT for that style. Be specific — describe layout, objects, labels, composition, colours. Mention concrete visual elements from the POST (e.g., if the post discusses grid tie inverter, mention a grid tie inverter in the image). Do NOT default to generic "Thai house with panels and a blue sky" unless the post is literally about a Thai house installation.

STYLE → PROMPT GUIDANCE (examples, not templates):
- `technical_diagram` — "Clean side-by-side technical diagram comparing X and Y, labelled boxes for inverter, battery, grid connection, arrows showing electricity flow, white background, flat vector style, Thai labels."
- `infographic` — "Vertical infographic showing 4 numbered steps of solar installation, each step with an icon, cost annotations in ฿, clean modern flat design."
- `illustration` — "Stylised cross-section of a silicon solar cell showing P-type and N-type layers, photons as arrows, clean educational style, soft colour palette."
- `photograph` — "Professional photograph of a modern Thai single-storey home with rooftop PV array, late-afternoon golden light, realistic, 4k."

OUTPUT FORMAT exactly (no JSON, no code fences, no extra prose):
Style: <one of: technical_diagram | infographic | illustration | photograph>
Prompt: <the 60–100 word prompt>"""
        return await self.chat(prompt)
