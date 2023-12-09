import re
def clean(code):
    result_text = re.findall(r"```([\w\W]*?)```\n", code)
    if len(result_text) > 0:
        result = result_text[0].replace("mermaid", "")
        return result
    else:
        return code


def mermaid(code: str) -> None:
    return f"""
        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """