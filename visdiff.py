import json
import html
import os
import sys
from uuid import uuid4

from jinja2 import Template


def main():
    with open(sys.argv[1]) as fin:
        diff = json.load(fin)
    src_before = diff["src_before"].split("\n")
    src_after = diff["src_after"].split("\n")
    offset_before = 0
    offset_after = 0
    
    salt = str(uuid4())
    ADD = "ADD:" + salt
    DEL = "DEL:" + salt
    MOD = "MOD:" + salt
    END = "END:" + salt

    def update_offset(pos, lines):
        line_offset = sum((len(line) + 1) for line in lines[:pos["line"] - 1])
        pos["offset"] = line_offset + pos["col"] - 1

    for edit in diff["script"]:
        action = edit[0]
        if action == "add":
            update_offset(edit[1], src_after)
            update_offset(edit[2], src_after)
        elif action == "delete":
            update_offset(edit[1], src_before)
            update_offset(edit[2], src_before)
        elif action == "modify":
            update_offset(edit[1]["after"][0], src_after)
            update_offset(edit[1]["after"][1], src_after)
            update_offset(edit[1]["before"][0], src_before)
            update_offset(edit[1]["before"][1], src_before)

    src_before = "\n".join(src_before)
    src_after = "\n".join(src_after)

    script = diff["script"]

    def before_key(e):
        if e[0] == "delete":
            return e[1]["line"], e[1]["col"]
        if e[0] == "add":
            return 1 << 32, 0
        if e[0] == "modify":
            return e[1]["before"][0]["line"], e[1]["before"][0]["col"]

    def after_key(e):
        if e[0] == "add":
            return e[1]["line"], e[1]["col"]
        if e[0] == "delete":
            return 1 << 32, 0
        if e[0] == "modify":
            return e[1]["after"][0]["line"], e[1]["after"][0]["col"]

    script.sort(key=after_key)

    pos_end = 0
    for edit in script:
        action = edit[0]
        if action == "add":
            pos_start = edit[1]["offset"]
            if pos_start < pos_end:
                pos_start = pos_end
            pos_end = edit[2]["offset"]
            if pos_start >= pos_end:
                pos_end = pos_start
                continue
            src_after = src_after[:pos_start + offset_after] + \
                ADD + \
                src_after[pos_start + offset_after:pos_end + offset_after] + \
                END + src_after[pos_end + offset_after:]
            offset_after += len(ADD + END)
        elif action == "delete":
            continue
        elif action == "modify":
            continue
            pos_start = edit[1]["after"][0]["offset"]
            if pos_start < pos_end:
                pos_start = pos_end
            pos_end = edit[1]["after"][1]["offset"]
            if pos_start >= pos_end:
                pos_end = pos_start
                continue
            src_after = src_after[:pos_start + offset_after] + \
                 MOD + \
                 src_after[pos_start + offset_after:pos_end + offset_after] + \
                 END + src_after[pos_end + offset_after:]
            offset_after += len(MOD + END)

    script.sort(key=before_key)

    pos_end = 0
    for edit in script:
        action = edit[0]
        if action == "delete":
            pos_start = edit[1]["offset"]
            if pos_start < pos_end:
                pos_start = pos_end
            pos_end = edit[2]["offset"]
            if pos_start >= pos_end:
                pos_end = pos_start
                continue
            src_before = src_before[:pos_start + offset_before] + \
                         DEL + \
                         src_before[pos_start + offset_before:pos_end + offset_before] + \
                         END + src_before[pos_end + offset_before:]
            offset_before += len(DEL + END)
        elif action == "add":
            continue
        elif action == "modify":
            continue
            pos_start = edit[1]["before"][0]["offset"]
            if pos_start < pos_end:
                pos_start = pos_end
            pos_end = edit[1]["before"][1]["offset"]
            if pos_start >= pos_end:
                pos_end = pos_start
                continue
            src_before = src_before[:pos_start + offset_before] + \
                 MOD + \
                 src_before[pos_start + offset_before:pos_end + offset_before] + \
                 END + src_before[pos_end + offset_before:]
            offset_before += len(MOD + END)

    src_before = html.escape(src_before)
    src_after = html.escape(src_after)

    def finalize(txt):
        return txt \
            .replace(ADD, "<span class=\"add\">") \
            .replace(DEL, "<span class=\"del\">") \
            .replace(MOD, "<span class=\"mod\">") \
            .replace(END, "</span>")

    src_before = finalize(src_before)
    src_after = finalize(src_after)

    with open("template.html") as fin:
        template = Template(fin.read())
    with open(os.path.splitext(sys.argv[1])[0] + ".html", "w") as fout:
        fout.write(template.render(before=src_before, after=src_after))


if __name__ == "__main__":
    sys.exit(main())