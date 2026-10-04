# Phase 2.0 — 投稿管道與格式對照(可投哪些 × 各自格式)

> **SSoT:** 本頁兩張表摘自並以 [`docs/references/quorum-paper-venues/quorum-paper-venues.md`](../../references/quorum-paper-venues/quorum-paper-venues.md)
> (2026-10-04 二次查證,每筆附官方來源)為準;本頁只保留與 Phase 2.0 撰稿決策直接相關的部分,並新增「認可度」欄。
> 截止日、費用、格式若有更新,**先改 reference 文件,再同步本頁**。
>
> **Phase 決策(使用者,2026-10-04):主投稿目標 = TACL**;失利則依序接 ARR → ACL 2027 / EMNLP 2027;TMLR 為認可度 Tier 2 的備案。
> 品質與完整度優先,不以 10/12(ARR)或 10/31(Industry)為目標。

---

## 表 1 · 可以投哪些(同一篇論文同一時間只能有一個典藏主家)

| 角色 | 選項 | 截止 | 認可度 Tier* | 可否並行 |
|---|---|---|---|---|
| **預印本** | arXiv cs.CL | 隨時(首投需個人 endorser) | — | ✅ 一律可並行;TACL 要求在 Comments to the Editor 揭露 |
| **典藏主家(擇一)** | ⭐ **TACL**(本 phase 主目標) | 每月 1 號 23:59 HST(下個 2026-11-01) | **1** | ❌ 與下列互斥 |
| | ARR → ACL 2027 | 2027-01-04 | **1**(main)/ 2(Findings 退路) | |
| | ARR → EMNLP 2027 | 未公布(暫定 2027-11 會議) | **1** | |
| | ARR → NAACL / COLING 2027 | 2026-10-12(**本 phase 不追**) | 1.5 / 2 | |
| | NAACL 2027 Industry Track | 2026-10-31(**本 phase 不追**) | 獨立 volume | |
| | TMLR | 隨時 | 2(證據弱) | |
| | Computational Linguistics | 隨時 | 1.5 | |
| **Demo(另一篇)** | NAACL 2027 System Demonstrations | 2026-12-04 | 獨立 volume | ⚠️ 主家為 TACL 時**不並行**(TACL 禁任何材料同時審查;Demo CFP 禁內容/結果大幅重疊)→ 主論文定案後投下一屆 Demo |
| **Workshop** | EACL / ICLR / NAACL 2027 workshops | ≈12/15、≈2/1、≈2/5(清單未公布) | 3 | ✅ 僅非典藏軌(且 TACL 視非典藏 workshop 為非典藏) |

\* **認可度 Tier** 的證據與來源見 reference 文件〈認可度排序〉一節(SSoT,不在此複製):Tier 1 = ACL、EMNLP、TACL;1.5 = NAACL、EACL、CL 期刊;2 = Findings、COLING、AACL、JAIR、TMLR;3 = Cambridge NLP、PeerJ CS。

---

## 表 2 · 各自要準備的格式

| 投稿處 | 模板 | 篇幅 | 匿名 | 額外要準備 |
|---|---|---|---|---|
| ⭐ **TACL**(主目標) | TACL LaTeX 官方樣式,**A4**;≥11 pt 正文 / ≥10 pt 圖說、行號、「Confidential TACL submission. DO NOT DISTRIBUTE.」頁首(格式不符可 desk reject) | 正文 **7–10 頁**,參考文獻不計;2024-03 起 appendix 不計入正文:重現細節 ≤5 頁、補充結果 ≤3 頁,**appendix 不審** → 主張所依賴的協定細節必須寫在正文 | **雙盲**:無作者/單位/致謝;自我引用改第三人稱;勿過度宣傳 | **不收補充材料、不得放任何補充材料連結(匿名也不行)** → 改以匿名文字說明程式/資料是否釋出;**投稿時即須提供所有作者 email、國家、單位**(並列於 Comments to the Editor);揭露 arXiv 等非典藏版本;經 TACL 自有系統投稿;不得同時在別處審查;免費 |
| **arXiv** | 不限(用去匿名的 TACL 版即可) | 不限 | 具名 | 個人 endorser(2026-01-21 新規);授權擇一且不可改(建議 CC BY 4.0) |
| ARR → ACL / EMNLP 2027(退路) | ACL/ARR LaTeX | 長文 8 頁 / 短文 4 頁;refs + appendix 不計 | 雙盲 | 必寫 Limitations;所有作者登記審稿;(Responsible NLP checklist 為 ARR 慣例,投前確認) |
| TMLR(備案) | TMLR LaTeX(OpenReview) | 無上限(>12 頁審稿 4 週) | 雙盲 | 不收會議論文擴充版 |
| NAACL Demo(主論文定案後的下一屆) | ACL 模板 | ≤6 頁(超過退稿),appendix ≤2 頁 | **單盲**(具名) | ≤2.5 分鐘錄影 + 可用 demo/套件 + 必須有評估;內容寫系統本身 |

**雙盲投稿的實務必備:** README、Hugging Face collection、Space、gated adapter、GitHub 皆會暴露身分。
- **TACL**:論文中**不得放任何程式/資料連結**(匿名 repo 也不行),只寫匿名的釋出聲明;審稿期間審稿人拿不到 adapter,論文須如實說明。公開 repo/Space/HF 頁面視為先前的非典藏版本,在 Comments to the Editor 揭露(系統名稱是否沿用 "Quorum" 為 must-ask)。
- **ARR / TMLR(退路)**:一般允許匿名補充材料/匿名連結(本 phase 未查證細節;改投時查證後再準備)。
