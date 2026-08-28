#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng GUI đơn giản: chọn thuật toán (WF5 / LSB / DCT / F5), ảnh cover, file bí mật,
thư mục + tên file ảnh stego; sau khi giấu tin hiển thị PSNR, SSIM, ER và điểm RS / Chi-square / SP.
Chạy từ thư mục gốc repo:  python stego_app.py
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# Cho phép import từ src/algorithms khi chạy từ thư mục gốc repo
_REPO = Path(__file__).resolve().parent
_ALGORITHMS = _REPO / "src" / "algorithms"
if str(_ALGORITHMS) not in sys.path:
    sys.path.insert(0, str(_ALGORITHMS))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np

import wf5
import lsb
import dct
import f5
import stego_scores


def _safe_join_output_dir_file(out_dir: str, filename: str) -> str:
    d = Path(out_dir).expanduser().resolve()
    name = (filename or "stego.png").strip()
    if not name.lower().endswith((".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")):
        name += ".png"
    return str(d / name)


def _verdict_from_avg(avg: float) -> str:
    if avg < 0.20:
        return "CLEAN — không thấy tín hiệu giấu tin rõ"
    if avg < 0.40:
        return "LOW — lệch nhẹ, có thể false positive"
    if avg < 0.60:
        return "MODERATE — đáng ngờ"
    return "HIGH — dấu hiệu giấu tin mạnh"


def embed_wf5_job(
    cover_path: str,
    msg_path: str,
    out_path: str,
    mu: float,
    x0: float,
    warmup: int,
    threshold: float,
) -> dict:
    cover = wf5.read_image_bgr(cover_path)
    with open(msg_path, "rb") as f:
        pt = f.read()
    ct_full = wf5.chaotic_xor(pt, mu, x0, warmup)
    cap_bits = int(wf5.max_capacity_bits(cover))
    total_img_bits = int(cover.size * 8)
    if cap_bits < 32:
        raise ValueError("Ảnh quá nhỏ: capacity < 32 bit (header).")
    max_ct_bytes = max(0, (cap_bits - 32) // 8)
    need_bits_full = 32 + len(ct_full) * 8
    too_large = need_bits_full > cap_bits
    ct_use = ct_full if not too_large else ct_full[:max_ct_bytes]
    payload_bits = wf5.build_payload_bits_from_ciphertext(ct_use)
    stego, nbits = wf5.embed_wf5(cover, payload_bits, threshold)
    if nbits != payload_bits.size:
        raise RuntimeError(f"Embed không đủ bit: {nbits} / {payload_bits.size}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wf5.write_image_bgr(out_path, stego)
    stego2 = wf5.read_image_bgr(out_path)
    psnr_v, ssim_v = wf5.compute_metrics(cover, stego2)
    er_img = nbits / total_img_bits if total_img_bits else 0.0
    er_cap = nbits / cap_bits if cap_bits else 0.0
    return {
        "algorithm": "WF5",
        "out_path": out_path,
        "psnr": psnr_v,
        "ssim": ssim_v,
        "ER_img": er_img,
        "ER_cap": er_cap,
        "nbits": nbits,
        "warn_truncation": too_large,
    }


def embed_lsb_job(
    cover_path: str,
    msg_path: str,
    out_path: str,
    mu: float,
    x0: float,
    warmup: int,
) -> dict:
    cover = wf5.read_image_bgr(cover_path)
    with open(msg_path, "rb") as f:
        pt = f.read()
    ct_full = lsb.chaotic_xor(pt, mu, x0, warmup)
    cap_bits = lsb.max_capacity_bits_lsb(cover)
    total_container_bits = int(cover.size * 8)
    if cap_bits < 32:
        raise ValueError("Ảnh quá nhỏ: capacity < 32 bit.")
    max_ct_bytes = max(0, (cap_bits - 32) // 8)
    truncated = len(ct_full) > max_ct_bytes
    ct = ct_full[:max_ct_bytes] if truncated else ct_full
    payload_bits = lsb.build_payload_bits_from_ct(ct)
    stego_img, nbits = lsb.embed_lsb(cover, payload_bits)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wf5.write_image_bgr(out_path, stego_img)
    stego = wf5.read_image_bgr(out_path)
    psnr_v, ssim_v = lsb.compute_metrics(cover, stego)
    er_img = nbits / total_container_bits if total_container_bits else 0.0
    er_cap = nbits / cap_bits if cap_bits else 0.0
    return {
        "algorithm": "LSB",
        "out_path": out_path,
        "psnr": psnr_v,
        "ssim": ssim_v,
        "ER_img": er_img,
        "ER_cap": er_cap,
        "nbits": nbits,
        "warn_truncation": truncated,
    }


def embed_dct_job(
    cover_path: str,
    msg_path: str,
    out_path: str,
    mu: float,
    x0: float,
    warmup: int,
    tau: float,
    pos_u: int,
    pos_v: int,
) -> dict:
    img = wf5.read_image_bgr(cover_path)
    with open(msg_path, "rb") as f:
        pt = f.read()
    cap = int(dct.max_capacity_bits(img))
    payload_bits_use, st = dct.build_payload_bits_truncate(pt, mu, x0, warmup, cap)
    if payload_bits_use.size == 0 and cap < 32:
        raise ValueError("Capacity < 32 bit, không thể nhúng header.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pos = (pos_u, pos_v)
    nbits = dct.embed_dct_full(cover_path, payload_bits_use, out_path, pos=pos, tau=tau)
    stego = wf5.read_image_bgr(out_path)
    psnr_v, ssim_v = dct.compute_metrics(img, stego)
    total_image_bits = int(img.size * 8)
    er_img = nbits / total_image_bits if total_image_bits else 0.0
    er_cap = nbits / cap if cap else 0.0
    return {
        "algorithm": "DCT",
        "out_path": out_path,
        "psnr": psnr_v,
        "ssim": ssim_v,
        "ER_img": er_img,
        "ER_cap": er_cap,
        "nbits": nbits,
        "dct_stats": st,
        "warn_truncation": st.get("completion_flag", 1) == 0,
    }


def embed_f5_job(
    cover_path: str,
    msg_path: str,
    out_path: str,
    mu: float,
    x0: float,
    warmup: int,
    passphrase: str,
    quality: int,
    k: int,
) -> dict:
    cover = wf5.read_image_bgr(cover_path)
    with open(msg_path, "rb") as f:
        pt = f.read()
    empty_payload = np.zeros(0, dtype=np.uint8)
    _, _, cap_bits, nonzero_ac = f5.embed_f5_like_dct(
        cover, empty_payload, passphrase, quality=quality, k=k
    )
    payload_bits, st = f5.build_payload_bits_trunc(pt, mu, x0, warmup, cap_bits)
    stego, embedded_bits, cap_bits2, _ = f5.embed_f5_like_dct(
        cover, payload_bits, passphrase, quality=quality, k=k
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wf5.write_image_bgr(out_path, stego)
    psnr_v, ssim_v = f5.compute_metrics(cover, stego)
    total_bits = int(cover.size * 8)
    er_img = embedded_bits / total_bits if total_bits else 0.0
    er_cap = embedded_bits / cap_bits2 if cap_bits2 else 0.0
    return {
        "algorithm": "F5",
        "out_path": out_path,
        "psnr": psnr_v,
        "ssim": ssim_v,
        "ER_message": st["ER_message"],
        "ER_img": er_img,
        "ER_cap": er_cap,
        "nbits": embedded_bits,
        "f5_stats": st,
        "nonzero_ac": nonzero_ac,
        "warn_truncation": st.get("complete_flag", 1) == 0,
    }


def format_report(embed_info: dict, scores: dict) -> str:
    lines = []
    lines.append(f"Thuật toán: {embed_info['algorithm']}")
    lines.append(f"File stego: {embed_info['out_path']}")
    lines.append("")
    lines.append("--- Chất lượng & tỷ lệ nhúng (cover vs stego) ---")
    lines.append(f"PSNR:  {embed_info['psnr']:.4f} dB")
    lines.append(f"SSIM:  {embed_info['ssim']:.6f}")
    if embed_info["algorithm"] == "F5":
        lines.append(
            f"ER (tin nhắn): {embed_info['ER_message']:.6f}  (bit ciphertext nhúng / tổng bit ciphertext)"
        )
        st = embed_info.get("f5_stats") or {}
        lines.append(
            f"  Cần {st.get('need_ct_bytes', '?')} B ciphertext, nhúng {st.get('use_ct_bytes', '?')} B"
        )
    lines.append(f"ER_img (bit nhúng / tổng bit ảnh): {embed_info['ER_img']:.8f}")
    lines.append(f"ER_cap (bit nhúng / capacity thuật toán): {embed_info['ER_cap']:.6f}")
    lines.append(f"Số bit payload (header+ct): {embed_info['nbits']}")
    if embed_info.get("warn_truncation"):
        lines.append("CẢNH BÁO: tin bị cắt bớt do vượt capacity.")
    lines.append("")
    lines.append("--- Phân tích mù (ảnh stego) — RS, Chi-square, Sample Pair ---")
    lines.append(f"RS_score (composite):    {scores['RS_score']:.6f}")
    lines.append(f"  R={scores['RS_R']:.4f} G={scores['RS_G']:.4f} B={scores['RS_B']:.4f}")
    lines.append(f"Chi2_score (composite):  {scores['Chi2_score']:.6f}")
    lines.append(f"  R={scores['Chi2_R']:.4f} G={scores['Chi2_G']:.4f} B={scores['Chi2_B']:.4f}")
    lines.append(f"SP_score (composite):    {scores['SP_score']:.6f}")
    lines.append(f"  R={scores['SP_R']:.4f} G={scores['SP_G']:.4f} B={scores['SP_B']:.4f}")
    lines.append(f"LSB entropy score:         {scores['LSB_score']:.6f}  (H={scores['LSB_H']:.6f})")
    lines.append(f"avg_score (trọng số):    {scores['avg_score']:.6f}")
    lines.append(f"Kết luận: {_verdict_from_avg(scores['avg_score'])}")
    lines.append("")
    lines.append("Raw (tham khảo):")
    lines.append(
        f"  RS_raw   R={scores['RS_raw_R']:.6f} G={scores['RS_raw_G']:.6f} B={scores['RS_raw_B']:.6f}"
    )
    lines.append(
        f"  Chi2_raw R={scores['Chi2_raw_R']:.6f} G={scores['Chi2_raw_G']:.6f} B={scores['Chi2_raw_B']:.6f}"
    )
    lines.append(
        f"  SP_raw   R={scores['SP_raw_R']:.6f} G={scores['SP_raw_G']:.6f} B={scores['SP_raw_B']:.6f}"
    )
    return "\n".join(lines)


class StegoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Steganography — WF5 / LSB / DCT / F5")
        self.geometry("780x640")
        self.minsize(640, 520)

        self.var_cover = tk.StringVar()
        self.var_secret = tk.StringVar()
        self.var_outdir = tk.StringVar()
        self.var_outname = tk.StringVar(value="stego_out.png")
        self.var_algo = tk.StringVar(value="WF5")
        self.var_mu = tk.StringVar(value="3.99")
        self.var_x0 = tk.StringVar(value="0.6")
        self.var_warmup = tk.StringVar(value="1000")
        self.var_wf5_threshold = tk.StringVar(value="30.0")
        self.var_dct_tau = tk.StringVar(value="5.0")
        self.var_dct_u = tk.StringVar(value="4")
        self.var_dct_v = tk.StringVar(value="3")
        self.var_f5_pass = tk.StringVar(value="abc123")
        self.var_f5_q = tk.StringVar(value="85")
        self.var_f5_k = tk.StringVar(value="3")

        pad = {"padx": 6, "pady": 4}
        r = 0
        frm = ttk.Frame(self, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        frm.rowconfigure(12, weight=1)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Thuật toán:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(
            frm,
            textvariable=self.var_algo,
            values=("WF5", "LSB", "DCT", "F5"),
            state="readonly",
            width=12,
        ).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        ttk.Label(frm, text="Ảnh cover:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_cover, width=55).grid(
            row=r, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Chọn…", command=self._pick_cover).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(frm, text="File bí mật:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_secret, width=55).grid(
            row=r, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Chọn…", command=self._pick_secret).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(frm, text="Thư mục lưu stego:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_outdir, width=55).grid(
            row=r, column=1, sticky="ew", **pad
        )
        ttk.Button(frm, text="Chọn…", command=self._pick_outdir).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(frm, text="Tên file ảnh stego:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_outname, width=55).grid(
            row=r, column=1, sticky="ew", **pad
        )
        ttk.Label(frm, text="(vd: Lenna_stego.png)").grid(row=r, column=2, sticky="w", **pad)
        r += 1

        sub = ttk.LabelFrame(frm, text="Tham số logistic (μ, x₀, warmup)", padding=6)
        sub.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(sub, text="μ").grid(row=0, column=0, padx=4)
        ttk.Entry(sub, textvariable=self.var_mu, width=10).grid(row=0, column=1, padx=4)
        ttk.Label(sub, text="x₀").grid(row=0, column=2, padx=4)
        ttk.Entry(sub, textvariable=self.var_x0, width=10).grid(row=0, column=3, padx=4)
        ttk.Label(sub, text="warmup").grid(row=0, column=4, padx=4)
        ttk.Entry(sub, textvariable=self.var_warmup, width=8).grid(row=0, column=5, padx=4)
        r += 1

        sub2 = ttk.LabelFrame(frm, text="Tham số theo thuật toán", padding=6)
        sub2.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(sub2, text="WF5 threshold:").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(sub2, textvariable=self.var_wf5_threshold, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(sub2, text="DCT τ / pos_u / pos_v:").grid(row=0, column=2, sticky="w", padx=8)
        ttk.Entry(sub2, textvariable=self.var_dct_tau, width=6).grid(row=0, column=3, padx=2)
        ttk.Entry(sub2, textvariable=self.var_dct_u, width=4).grid(row=0, column=4, padx=2)
        ttk.Entry(sub2, textvariable=self.var_dct_v, width=4).grid(row=0, column=5, padx=2)
        ttk.Label(sub2, text="F5 passphrase / Q / k:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(sub2, textvariable=self.var_f5_pass, width=14).grid(row=1, column=1, padx=4, pady=4)
        ttk.Entry(sub2, textvariable=self.var_f5_q, width=5).grid(row=1, column=3, padx=2, pady=4)
        ttk.Entry(sub2, textvariable=self.var_f5_k, width=4).grid(row=1, column=4, padx=2, pady=4)
        r += 1

        ttk.Button(frm, text="Giấu tin & tính chỉ số", command=self._run_embed).grid(
            row=r, column=0, columnspan=3, pady=10
        )
        r += 1

        ttk.Label(frm, text="Kết quả:").grid(row=r, column=0, sticky="nw", **pad)
        self.txt = scrolledtext.ScrolledText(frm, height=22, width=88, font=("Consolas", 9))
        self.txt.grid(row=r, column=1, columnspan=2, sticky="nsew", **pad)
        frm.rowconfigure(r, weight=1)

    def _pick_cover(self) -> None:
        p = filedialog.askopenfilename(
            title="Chọn ảnh cover",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All", "*.*"),
            ],
        )
        if p:
            self.var_cover.set(p)

    def _pick_secret(self) -> None:
        p = filedialog.askopenfilename(title="Chọn file bí mật", filetypes=[("All", "*.*")])
        if p:
            self.var_secret.set(p)

    def _pick_outdir(self) -> None:
        p = filedialog.askdirectory(title="Chọn thư mục lưu ảnh stego")
        if p:
            self.var_outdir.set(p)

    def _parse_floats(self) -> tuple[float, float, int, float, int, int, int, str, int, int]:
        mu = float(self.var_mu.get().strip())
        x0 = float(self.var_x0.get().strip())
        warmup = int(self.var_warmup.get().strip())
        tau = float(self.var_dct_tau.get().strip())
        pos_u = int(self.var_dct_u.get().strip())
        pos_v = int(self.var_dct_v.get().strip())
        thr = float(self.var_wf5_threshold.get().strip())
        passphrase = self.var_f5_pass.get().strip() or "abc123"
        quality = int(self.var_f5_q.get().strip())
        k = int(self.var_f5_k.get().strip())
        return mu, x0, warmup, tau, pos_u, pos_v, thr, passphrase, quality, k

    def _run_embed(self) -> None:
        cover = self.var_cover.get().strip()
        secret = self.var_secret.get().strip()
        outdir = self.var_outdir.get().strip()
        outname = self.var_outname.get().strip()
        algo = self.var_algo.get().strip()

        if not cover or not os.path.isfile(cover):
            messagebox.showerror("Lỗi", "Chọn file ảnh cover hợp lệ.")
            return
        if not secret or not os.path.isfile(secret):
            messagebox.showerror("Lỗi", "Chọn file bí mật hợp lệ.")
            return
        if not outdir or not os.path.isdir(outdir):
            messagebox.showerror("Lỗi", "Chọn thư mục lưu ảnh stego hợp lệ.")
            return

        try:
            mu, x0, warmup, tau, pos_u, pos_v, thr, passphrase, quality, k = self._parse_floats()
        except ValueError as e:
            messagebox.showerror("Lỗi", f"Tham số số không hợp lệ: {e}")
            return

        out_path = _safe_join_output_dir_file(outdir, outname)
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "Đang xử lý…\n")
        self.update_idletasks()

        def work() -> None:
            try:
                if algo == "WF5":
                    info = embed_wf5_job(cover, secret, out_path, mu, x0, warmup, thr)
                elif algo == "LSB":
                    info = embed_lsb_job(cover, secret, out_path, mu, x0, warmup)
                elif algo == "DCT":
                    info = embed_dct_job(cover, secret, out_path, mu, x0, warmup, tau, pos_u, pos_v)
                elif algo == "F5":
                    info = embed_f5_job(
                        cover, secret, out_path, mu, x0, warmup, passphrase, quality, k
                    )
                else:
                    raise ValueError(f"Thuật toán không hỗ trợ: {algo}")

                scores = stego_scores.compute_scores(out_path)
                text = format_report(info, scores)
                self.after(0, lambda t=text: self._show_result(t, None))
            except Exception as e:
                err_s = str(e)
                self.after(0, lambda err=err_s: self._show_result(None, err))

        threading.Thread(target=work, daemon=True).start()

    def _show_result(self, text: str | None, err: str | None) -> None:
        self.txt.delete("1.0", tk.END)
        if err:
            self.txt.insert(tk.END, f"Lỗi:\n{err}\n")
            messagebox.showerror("Lỗi", err)
        else:
            self.txt.insert(tk.END, text or "")


def main() -> None:
    app = StegoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
