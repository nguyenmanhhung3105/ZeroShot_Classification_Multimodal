"""Dựng sơ đồ tài liệu từ cấu trúc cố định; không import code hoặc mở dữ liệu.

Chạy: python3 docs/so_do_he_thong/so_do/tao_so_do.py
Yêu cầu Graphviz (lệnh dot). Nguồn DOT và ảnh SVG được đặt cạnh script.
"""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
COLORS = {
    "input": "#dbeafe", "process": "#e2e8f0", "model": "#ede9fe",
    "score": "#fef3c7", "output": "#dcfce7", "control": "#ffedd5",
}


def diagram(name, title, nodes, edges, ranks=()):
    """Node: (id, nhãn, loại); edge: (nguồn, đích[, nhãn[, nét đứt]])."""
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    lines = ["digraph G {",
             'graph [rankdir=TB, bgcolor="white", fontname="Arial", fontsize=19,',
             '  pad=0.3, nodesep=0.45, ranksep=0.65, splines=polyline,',
             f'  labelloc=t, label={quote(title)}];',
             'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=12,',
             '  color="#64748b", fontcolor="#0f172a", penwidth=1.2,',
             '  margin="0.18,0.12", width=2.65];',
             'edge [fontname="Arial", fontsize=10, color="#475569",',
             '  fontcolor="#334155", arrowsize=0.75];']
    identifiers = {node[0] for node in nodes}
    if len(identifiers) != len(nodes):
        raise ValueError(f"Node trùng trong {name}")
    for identifier, label, kind in nodes:
        lines.append(f'{identifier} [label={quote(label)}, fillcolor="{COLORS[kind]}"];')
    for edge in edges:
        source, target, *options = edge
        if source not in identifiers or target not in identifiers:
            raise ValueError(f"Edge không có node trong {name}: {edge}")
        attrs = []
        if options and options[0]:
            attrs.append("label=" + quote(options[0]))
        if len(options) > 1 and options[1]:
            attrs.append('style="dashed"')
        lines.append(f'{source} -> {target}' + (' [' + ', '.join(attrs) + ']' if attrs else '') + ';')
    for rank in ranks:
        if not set(rank) <= identifiers:
            raise ValueError(f"Rank không có node trong {name}")
        lines.append('{rank=same; ' + '; '.join(rank) + ';}')
        # Giữ thứ tự trái → phải; rank=same riêng lẻ chưa khóa thứ tự các ô.
        for left, right in zip(rank, rank[1:]):
            lines.append(f'{left} -> {right} [style=invis, weight=100];')
    lines.append('}')
    source = ROOT / (name + '.dot')
    source.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    subprocess.run(["dot", "-Tsvg", str(source), "-o", str(ROOT / (name + '.svg'))], check=True)


def main():
    if not shutil.which("dot"):
        raise SystemExit("Cần Graphviz: không tìm thấy lệnh dot")
    diagram("01_tong_quan", "Ba luồng chạy độc lập của dự án", [
        ("cfg", "CẤU HÌNH + TAXONOMY\nẢnh và text của mẫu", "input"),
        ("main", "HỆ THỐNG CHÍNH\nrun_experiment.py", "process"),
        ("up", "NGHIÊN CỨU UPSTREAM\nrun_upstream.py", "process"),
        ("ad", "NGHIÊN CỨU ADAPTED\nrun_adapted_methods.py", "process"),
        ("m", "MetaCLIP2 / DFN5B / SigLIP2\nPrompt dự án", "model"),
        ("u", "OpenAI CLIP\nHàm prompt/scoring tác giả", "model"),
        ("a", "MetaCLIP2 / DFN5B / SigLIP2\nThay prompt nhánh ảnh", "model"),
        ("mo", "Ảnh + text → fusion\nNhãn + log + metric", "output"),
        ("uo", "original_image / multimodal\nNhãn + log + metric", "output"),
        ("ao", "Ảnh + text → fusion\nĐối chứng phương pháp/seed", "output"),
    ], [("cfg","main"),("cfg","up"),("cfg","ad"),("main","m"),("up","u"),
        ("ad","a"),("m","mo"),("u","uo"),("a","ao")],
        [("main","up","ad"),("m","u","a"),("mo","uo","ao")])
    diagram("02_pipeline", "Hệ thống chính: input → suy luận → output", [
        ("cfg", "YAML\nModel, task, prompt, trọng số", "control"),
        ("data", "BẢNG DỮ LIỆU\nID + text + đường dẫn ảnh", "input"),
        ("load", "Đọc giới hạn + lọc text\nKiểm tra cột và ảnh hợp lệ", "process"),
        ("prompt", "PROMPT CÁC LỚP\nTaxonomy của task", "input"),
        ("img", "Ảnh RGB theo batch\nPreprocess → image encoder", "model"),
        ("text", "Text mẫu / plot\nTokenizer → text encoder", "model"),
        ("cls", "Prompt → text encoder\nmean_embed / mean_sim", "model"),
        ("score", "Điểm ảnh–prompt và text–prompt\nHai bảng điểm N × C", "score"),
        ("pair", "Chỉ khi flag enabled=true\nẢnh ↔ text cùng mẫu\nCosine; một nhãn thêm JS/entropy", "score"),
        ("decision", "Softmax hoặc relative_z/sigmoid\nFusion → quan hệ nếu class_aware\n→ quy tắc chọn nhãn", "score"),
        ("pred", "NHÃN DỰ ĐOÁN\nMột nhãn / danh sách thể loại", "output"),
        ("truth", "NHÃN THẬT\nTừ cột nhãn của task", "input"),
        ("eval", "Metric + audit + lưu kết quả\nTSV / JSON / summary", "output"),
    ], [("cfg","load","cấu hình",True),("cfg","prompt","task",True),
        ("data","load"),("load","img"),("load","text"),("prompt","cls"),
        ("img","score"),("text","score"),("cls","score"),("score","decision"),
        ("img","pair"),("text","pair"),("pair","decision","policy / diagnostic"),
        ("decision","pred"),("pred","eval"),("truth","eval","đánh giá",True)],
        [("data","prompt"),("img","text","cls"),("score","pair"),("pred","truth")])
    for name, title, arch, tokenizer in [
        ("03_metaclip2", "MetaCLIP2: ba đường mã hóa", "ViT-H-14-worldwide-quickgelu", "Tokenizer theo kiến trúc"),
        ("04_dfn5b", "DFN5B: ba đường mã hóa", "apple/DFN5B-CLIP-ViT-H-14", "Tokenizer ViT-H-14"),
        ("05_siglip2", "SigLIP2: ba đường mã hóa", "ViT-SO400M-14-SigLIP2", "Tokenizer theo kiến trúc"),
    ]:
        diagram(name, title, [
            ("i", "ẢNH RGB", "input"),("t", "TEXT MẪU", "input"),("p", "PROMPT TỪNG LỚP", "input"),
            ("ip", "Preprocess của checkpoint\nResize/crop/normalize theo model", "process"),
            ("tt", tokenizer + "\ncontext_length của model", "process"),
            ("pt", tokenizer + "\ncontext_length của model", "process"),
            ("ie", "IMAGE ENCODER\n" + arch.replace("-worldwide-quickgelu", ""), "model"),
            ("te", "TEXT ENCODER\nCùng trọng số với prompt", "model"),
            ("pe", "TEXT ENCODER\nCùng trọng số với text mẫu", "model"),
            ("iv", "L2 → I [N, D]", "process"),("tv", "L2 → T [N, D]\nPlot dài: xem chunking", "process"),
            ("pv", "L2 từng prompt → gộp theo lớp\nE [C, D]", "process"),
            ("s", "I × Eᵀ và T × Eᵀ\nScale/bias hoặc relative_z\nFusion + quan hệ theo config\n→ chọn nhãn", "score"),
            ("o", "NHÃN + ĐIỂM TỪNG NHÁNH\nMetric / TSV / JSON", "output"),
        ], [("i","ip"),("t","tt"),("p","pt"),("ip","ie"),("tt","te"),("pt","pe"),
            ("ie","iv"),("te","tv"),("pe","pv"),("iv","s"),("tv","s"),("pv","s"),("s","o")],
            [("i","t","p"),("ip","tt","pt"),("ie","te","pe"),("iv","tv","pv")])
    diagram("06_single_label", "Một nhãn: hiệu chỉnh từng nhánh rồi fusion", [
        ("i", "S_img = I × E_imgᵀ\nCosine / mean cosine", "input"),
        ("t", "S_text = T × E_textᵀ\nCosine / mean cosine", "input"),
        ("il", "L_img = s_model × S_img + b\nScale/bias từ checkpoint", "score"),
        ("tl", "L_text = s_text × S_text + b\nScale riêng hoặc scale model", "score"),
        ("ip", "P_img = softmax(L_img)\nQua C lớp của từng mẫu", "score"),
        ("tp", "P_text = softmax(L_text)\nQua C lớp của từng mẫu", "score"),
        ("f", "P = w × P_img + (1-w) × P_text\nText rỗng: w hiệu lực = 1", "score"),
        ("rel", "Quan hệ nếu flag bật\nFakeddit 6way: class_aware\nTask khác: diagnostic", "process"),
        ("o", "argmax theo lớp\nMỗi mẫu đúng một nhãn", "output"),
    ], [("i","il"),("t","tl"),("il","ip"),("tl","tp"),("ip","f"),("tp","f"),("f","rel"),("rel","o")],
        [("i","t"),("il","tl"),("ip","tp")])
    diagram("07_multi_label", "MM-IMDb: hai chế độ quyết định nhiều nhãn", [
        ("s", "S_img và S_text [N, 23]\nCosine / mean cosine", "input"),
        ("z", "relative_z — ĐANG DÙNG\nZ-score từng mẫu, từng nhánh\nBỏ scale, bias, sigmoid", "score"),
        ("p", "fixed_threshold — TÙY CHỌN\nScale/bias rồi sigmoid\nTừng nhãn độc lập", "score"),
        ("zf", "Fusion điểm z\nText rỗng: dùng ảnh", "score"),
        ("pf", "Fusion điểm sigmoid\nText rỗng: dùng ảnh", "score"),
        ("zd", "Chọn điểm > z_threshold\nHiện tại: > 1.0", "process"),
        ("pd", "Chọn điểm ≥ threshold\nCấu hình sẵn: ≥ 0.22", "process"),
        ("fb", "Không có nhãn vượt ngưỡng?\nfallback_top1=true → argmax\nfalse → danh sách rỗng", "process"),
        ("o", "DANH SÁCH THỂ LOẠI\nScore + cờ fallback + metric", "output"),
    ], [("s","z"),("s","p"),("z","zf"),("p","pf"),("zf","zd"),("pf","pd"),
        ("zd","fb"),("pd","fb"),("fb","o")],[("z","p"),("zf","pf"),("zd","pd")])
    diagram("08_chunking", "Plot dài: chia đoạn trước text encoder", [
        ("in", "PLOT KHÔNG RỖNG\nTokenizer + context_length thực tế", "input"),
        ("split", "Đếm cả token đặc biệt\nChia theo câu → từ → ký tự\nMỗi đoạn vừa context", "process"),
        ("enc", "TEXT ENCODER + L2\nTừng batch các đoạn", "model"),
        ("mean", "mean_embed — ĐANG DÙNG\nTrung bình vector các đoạn\nChuẩn hóa L2 lại", "process"),
        ("max", "max_sim — TÙY CHỌN\nCosine từng đoạn với từng lớp\nLấy max riêng từng lớp", "score"),
        ("ms", "T_mean × E_textᵀ\nMột bảng điểm N × C", "score"),
        ("mx", "Điểm max [N, C]\nKhông có một vector T chung", "score"),
        ("out", "relative_z hoặc scale/sigmoid\nFusion + chọn thể loại", "output"),
    ], [("in","split"),("split","enc"),("enc","mean"),("enc","max"),("mean","ms"),
        ("max","mx"),("ms","out"),("mx","out")],[("mean","max"),("ms","mx")])
    diagram("09_openai_clip", "OpenAI CLIP gốc: ảnh so với prompt tên lớp", [
        ("i", "ẢNH RGB", "input"),("p", "TÊN LỚP + TEMPLATE", "input"),
        ("ip", "Resize bicubic + center crop\nTensor + normalize CLIP", "process"),
        ("pt", "SimpleTokenizer / BPE\nSOT + tokens + EOT + padding\nContext = 77", "process"),
        ("iv", "ViT: patch → transformer\nToken lớp → projection\nRN: ResNet → attention pool", "model"),
        ("tv", "Token + positional embedding\nCausal transformer\nVector EOT → projection", "model"),
        ("in", "L2 embedding ảnh", "process"),("tn", "L2 embedding prompt", "process"),
        ("s", "Cosine ảnh–prompt [N, C]\nargmax giữ nhãn cao nhất", "score"),
        ("o", "NHÃN PHÂN LOẠI ẢNH\nTrong dự án: log + metric", "output"),
    ],[("i","ip"),("p","pt"),("ip","iv"),("pt","tv"),("iv","in"),("tv","tn"),
        ("in","s"),("tn","s"),("s","o")],[("i","p"),("ip","pt"),("iv","tv"),("in","tn")])
    diagram("10_waffleclip", "WaffleCLIP: nhiều prompt ngẫu nhiên cho mỗi lớp", [
        ("classes", "TÊN CÁC LỚP + TEMPLATE\nSeed + waffle_pairs", "input"),
        ("random", "Từ điển word_list.pkl\nKý tự từ mô tả đầu vào\nSinh cặp từ / ký tự", "process"),
        ("prompts", "Cùng suffix ngẫu nhiên giữa lớp\nĐổi tên lớp trong prompt\n15 cặp → 30 prompt / lớp", "process"),
        ("te", "CLIP TEXT ENCODER\nL2 từng prompt → V_c [K_c, D]", "model"),
        ("img", "ẢNH → CLIP IMAGE ENCODER\nL2 → I [N, D]", "model"),
        ("cos", "I × V_cᵀ\nĐiểm từng ảnh × từng prompt", "score"),
        ("mean", "mean điểm qua K_c prompt\nS_img [N, C]", "score"),
        ("out", "GỐC: argmax → nhãn ảnh\nDỰ ÁN: có thể fusion với text", "output"),
    ],[("classes","random"),("random","prompts"),("prompts","te"),("te","cos"),
        ("img","cos"),("cos","mean"),("mean","out")],[("te","img")])
    diagram("11_dclip", "DCLIP: nhận diện bằng mô tả của từng lớp", [
        ("names", "TÊN LỚP", "input"),("desc", "DESCRIPTOR CỐ ĐỊNH\nPaper: sinh trước bằng LLM\nDự án: JSON viết từ taxonomy", "input"),
        ("render", "Tên lớp + mô tả → câu prompt\nTemplate theo hparams\nMỗi lớp có K_c mô tả", "process"),
        ("text", "CLIP TEXT ENCODER\nL2 từng prompt → V_c", "model"),
        ("image", "ẢNH → CLIP IMAGE ENCODER\nL2 → I", "model"),
        ("s", "I × V_cᵀ → mean qua mô tả\nS_img [N, C]", "score"),
        ("o", "GỐC: argmax → nhãn ảnh\nDỰ ÁN: có thể fusion với text", "output"),
    ],[("names","render"),("desc","render"),("render","text"),("text","s"),
        ("image","s"),("s","o")],[("names","desc"),("text","image")])
    diagram("12_upstream", "Runner upstream: code tác giả + dữ liệu dự án", [
        ("cfg", "upstream.yaml + base_config\nRepo ghim commit + descriptors\nFlag quan hệ mặc định false", "control"),
        ("plan", "Kiểm tra repo / taxonomy\nSinh và kiểm token prompt\nLưu frozen_plan.json", "process"),
        ("prep", "--prepare-only?\nCó → dừng tại plan\nKhông → tiếp tục", "process"),
        ("input", "Loader chung → cohort hợp lệ\nOpenAI CLIP checkpoint", "input"),
        ("encode", "Mã hóa ảnh + text một lần\nMỗi backbone/cohort\nTái dùng giữa method/seed", "model"),
        ("score", "clip / waffle / dclip_labels / dclip\nHàm mã hóa + mean của tác giả\nS_img và S_text chung", "score"),
        ("pair", "Flag quan hệ = true?\nCó: cosine cặp; JS/entropy tùy task\nKhông: giữ scoring nền", "process"),
        ("native", "original_image\nw = 1; quan hệ chỉ diagnostic\nMM-IMDb: quy tắc mở rộng", "process"),
        ("multi", "multimodal\nScale hoặc relative_z → fusion\nQuan hệ theo taxonomy nếu bật", "process"),
        ("out", "Nhãn + metric + audit từng trial\nsummary.tsv / seed_summary.tsv\nraw_predictions/", "output"),
    ],[("cfg","plan"),("plan","prep"),("prep","input","tiếp tục"),("input","encode"),
        ("encode","score"),("score","pair"),("pair","native"),("pair","multi"),("native","out"),("multi","out")],
        [("native","multi")])
    diagram("13_adapted", "Runner adapted: thay prompt ảnh, giữ nhánh text", [
        ("cfg", "adapted.yaml + base_config\nMethod + seeds + taxonomy\nFlag quan hệ mặc định false", "control"),
        ("plan", "Chuẩn bị prompt ảnh từng trial\nGiữ prompt text nền\nLưu frozen_plan.json", "process"),
        ("prep", "--prepare-only?\nCó → dừng tại plan\nKhông → tải model", "process"),
        ("model", "MetaCLIP2 / DFN5B / SigLIP2\nModel trong base_config", "model"),
        ("runner", "Runner dataset của dự án\nCohort giống nhau giữa trial", "process"),
        ("img", "NHÁNH ẢNH\nE_img: baseline / class_name\ncurrent_mean_sim / waffle / dclip", "score"),
        ("text", "NHÁNH TEXT\nE_text: prompt nền giữ chung\nAggregation theo base_config", "score"),
        ("fuse", "Scoring + fusion dự án\nQuan hệ nếu flag bật\nKhác cohort → dừng", "process"),
        ("out", "Nhãn + log + metric\nsummary.tsv / seed_summary.tsv\nraw_predictions/", "output"),
    ],[("cfg","plan"),("plan","prep"),("prep","model","tiếp tục"),("model","runner"),
        ("runner","img"),("runner","text"),("img","fuse"),("text","fuse"),("fuse","out")],
        [("img","text")])
    diagram("14_du_lieu_output", "Cohort và output: nhãn thật chỉ đi vào đánh giá", [
        ("in", "TSV / CSV / Parquet processed\nẢnh + text + nhãn thật", "input"),
        ("sel", "Lọc text → lấy max_samples\nKiểm tra ảnh → bỏ ảnh lỗi\nKhông bù mẫu tiếp theo", "process"),
        ("co", "COHORT HỢP LỆ\nCùng ID, text, ảnh, label_order", "process"),
        ("infer", "ẢNH + TEXT + PROMPT\nMã hóa → chấm điểm → nhãn", "model"),
        ("truth", "NHÃN THẬT\nNormalize / parse genres", "input"),
        ("metric", "So dự đoán với nhãn thật\nF1 / accuracy / audit nhánh", "process"),
        ("log", "raw_predictions.tsv\nerror_analysis.tsv\nmanifest.json", "output"),
        ("audit", "diagnostics.tsv\ntext_lengths.tsv\nevaluation_snapshot.json", "output"),
        ("sum", "Summary tổng hợp\nNghiên cứu: thêm seed_summary\nKhông chọn seed tốt nhất", "output"),
    ],[("in","sel"),("sel","co"),("co","infer"),("co","truth"),("infer","metric"),
        ("truth","metric","đánh giá",True),("metric","log"),("metric","audit"),("metric","sum")],
        [("infer","truth"),("log","audit","sum")])
    diagram("15_quan_he_anh_text", "Ảnh–text (khi flag bật): hiệu chỉnh theo nhãn ứng viên", [
        ("pair", "I và T của cùng mẫu\nL2; plot: mean các chunk", "input"),
        ("probs", "P_img và P_text [N, C]\nSoftmax riêng từng nhánh", "input"),
        ("cos", "Cosine r = I · T\nm = (1+r)/2; d = (1-r)/2", "score"),
        ("js", "JS chuẩn hóa → J ∈ [0,1]\nEntropy hai nhánh → R ∈ [0,1]\nPhẳng → R = 0", "score"),
        ("ev", "A = m × (1-J) × R\nD = d × J × R", "score"),
        ("base", "B = w P_img + (1-w) P_text\nĐiểm fusion nền", "score"),
        ("policy", "POLICY CỐ ĐỊNH THEO TAXONOMY\n0=True: e=A; 2=False Connection: e=D\n1/3/4/5: e=0, neutral", "control"),
        ("out", "U_c = B_c × (1 + 0.25 e_c)\nChuẩn hóa tổng → P' → argmax\nThiếu text / diagnostic → giữ B", "output"),
    ],[("pair","cos"),("probs","js"),("cos","ev"),("js","ev"),("probs","base"),
        ("ev","policy"),("policy","out"),("base","out")],[("pair","probs"),("cos","js")])
    print("Đã dựng 15 sơ đồ DOT + SVG; không truy cập dataset hoặc model.")


if __name__ == "__main__":
    main()
