# Quorum 這篇論文,該投去哪?——學術投稿管道全盤整理
## Where should the Quorum paper go? — A venue guide for submission (journals / conferences / workshops)

> **狀態 / As of:** 2026-10-04(同日二次查證:原「待查證清單」已逐項對照官方頁面確認並寫回本文,見文末〈查證紀錄〉)。標注【已查證】者皆附官方來源;官方頁面本身未公布的,標為【官方未公布】而非猜測。投稿方案與費用變動快,送出前請再看一次官方 CFP。
> Re-verified on 2026-10-04: every item from the former "to-verify" list was checked against official pages and folded back in (see the verification log at the end). Items the official pages don't state are marked **[not stated officially]** rather than guessed.

---

## 前言 · 這篇論文到底在賣什麼(決定它適合投哪) · What this paper is (and why that dictates the venue)

Quorum 是一個「小型開源模型的陪審團」,做 **calibration-free 的 zero-shot(與 24-shot)意圖分類**。要選投稿管道,重點不是「它做意圖分類」,而是**它的貢獻長什麼樣子**——因為是貢獻型態、不是任務,決定了哪個 venue 家族會收。

Quorum is "a jury of tiny open models" for **calibration-free zero-shot (and 24-shot) intent classification**. What picks the venue is **the _shape_ of the contribution**, not the task:

| 貢獻 / Contribution | 對應的投稿家族 / Maps to venue family |
|---|---|
| ① 「集成穩定打敗自己最強的單一成員」,6 個資料集 + **paired McNemar 顯著性** / a cheap ensemble reliably beats its own best member, with paired significance tests | 主會議 / Findings / TMLR / TACL(紮實的實證貢獻) |
| ② 三個 <1B 模型、CPU ~120ms、無訓練無校準 / tiny models, CPU-only, no training/calibration | **高效/綠色 NLP**(SustaiNLP、ENLSP——目前皆停辦)、On-device/efficient ML workshops |
| ③ 可量測的**負面結果** + 「五個抓到自己出錯的地方」自我批判 / measured negative results + methodological self-critique | **Insights from Negative Results**、ICBINB、Eval4NLP;期刊中 **Computational Linguistics** 明文歡迎 |
| ④ 嚴謹、可重現、全開源 MIT / rigorous, reproducible, fully open-source | **TMLR**(只看「證據是否充分 + 是否有人感興趣」)、TACL。⚠️ **ReScience C 不適用**(不收作者自己的研究,見清單 D) |
| ⑤ 現場 demo + 可自架 FastAPI 服務、勝過「重現版閉源商用 API」 / live demo + self-hostable service, beats a reproduced closed commercial API | **System Demonstrations track**、**Industry Track** |

→ 結論:**這篇論文可以餵養好幾個 venue 家族**(同儕審查主家 1 個 + 條件式的 Demo 1 篇 + 非典藏 workshop 摘要 + arXiv preprint)。下面先講「能不能一稿多投」的規則,再給時間軸與完整清單。

---

## ⚠️ 先搞懂:能「同時投多個」嗎?一稿多投規則 · Can you submit to several at once? The dual-submission rules

可以並行,但有硬規則,**踩線會被 desk-reject**(以下引文皆已查證):

You _can_ pursue several venues in parallel, but within hard rules (violating them = desk reject). All quotes verified:

1. **同一篇完整論文,同一時間只能有「一個同儕審查的典藏之家」。** 走會議(ARR)或走期刊,擇一。
   - TACL:「no material in any paper submitted to TACL may be under review (or published) at another journal, conference, or archival workshop venue at any time while it is under consideration by TACL.」【[src](https://transacl.org/index.php/tacl/about/submissions)】
   - TMLR:不得與「published, accepted for publication, or submitted in parallel at another archival, peer-reviewed venue」的論文重用文字/圖/結果;且「**We do not accept submissions that are expanded versions of conference papers.**」【[src](https://jmlr.org/tmlr/editorial-policies.html)】→ 若先在會議發表,之後就不能把擴充版投 TMLR。
2. **主會議 vs Industry Track 互斥**(同一篇只能投其一),依想強調「實證嚴謹」或「落地應用」擇一框架。
3. **System Demonstrations:可以「另寫一篇」,但有條件。** NAACL 2027 Demo CFP:「We will not consider any submissions that **overlap substantially in content or results** with papers being reviewed or already published elsewhere.」【[src](https://2027.naacl.org/calls/system_demonstration/)】
   - → Demo 論文必須聚焦**系統本身**(架構、API、部署、延遲、使用流程、可自架),**不可大幅重用主論文的實驗內容/結果**;引用主論文的 arXiv 版即可。
   - → **走 TACL 時風險最高**(TACL 禁止「任何材料」同時在別處審查)。若主家選 TACL,建議 **Demo 不並行**,等主論文定案後再投下一屆 Demo。走 ARR / TMLR 時,只要 Demo 內容清楚區隔,一般可行。
4. **非典藏(non-archival)workshop 可與「別處已發表/審查中的完整論文」並存。** 例:Insights 收「1-2 page non-archival abstract submissions for papers published elsewhere」;ICBINB「there will always be an option for authors to have their paper in a non-archival track」。TACL/TMLR 也明說非典藏 workshop 不算典藏發表。
5. **arXiv preprint 任何時候都可以。**
   - TACL:「Preprint servers such as arXiv.org … are _not_ considered archival」,但投稿時須在 Comments to the Editor 揭露 preprint 資訊。
   - TMLR:與「preprint servers such as arXiv」重疊是允許的。
   - 先上 arXiv 卡時間戳與可引用性,不影響後續投稿。

> **合法並行組合 / The legal parallel bundle:**
> `arXiv 先行` + `一個典藏主家(會議 ARR 其一 ‖ 期刊其一)` + `(條件式)一篇內容區隔的 System Demo` + `若干非典藏 workshop 摘要`。

---

## 先講結論 · TL;DR 推薦路線 · Recommended plan

> **今天(2026-10-04)的現實:2026 會議年的投稿幾乎全數截止。** 還開著的「活」窗口在下面;最對味的 thematic workshops(SustaiNLP、ENLSP、NLP4ConvAI)目前停辦,2027 workshop 清單也還沒公布(提案已於 10/2 放榜,清單應很快出爐)。
> **Reality check:** almost every 2026-cycle deadline has passed. Live windows below; the best-fit thematic workshops are dormant, and the 2027 workshop lists aren't out yet.

**建議執行順序 / Do this, in order:**

1. **現在就上 arXiv(cs.CL)** — 免費、可引用、卡時間戳。**注意 2026-01-21 起 endorsement 規則變嚴**:首次投 cs 類別、且沒有既有 arXiv 論文的人,即使有學校信箱也**需要一位現役作者個人 endorse**——先去約。
2. **主家二選一(不可並行):**
   - **路線 A(會議):** 完整論文投 **ARR 2026 年 10 月 cycle(10/12)** → commit **NAACL 2027**(舊金山,6/1–5,可能被收為 Main 或 Findings)或 **COLING 2027**(澳門,5/9–14,只有 Main)。**錯過 10/12 的話**:下一個是 **ARR 2027 年 1 月 cycle(2027-01-04)→ ACL 2027**(京都,8/17–22,未入選 main 者自動考慮 Findings)。
   - **路線 B(期刊):** 投 **TMLR**(免費、無截止、明文「novelty 非必要」、雙盲)或 **TACL**(免費、**下個截止 2026-11-01 23:59 夏威夷時間**、雙盲、NLP 原生)。獨立作者、無會議死線壓力時最舒服。
3. **條件式加投 System Demo:** **NAACL 2027 System Demonstrations(直投,11/1 開放投稿,12/4 截止,單盲)**——你已有現場 demo + 可自架服務 + 可錄影 + 已量測;但 Demo 論文**內容須與主論文區隔**(見規則 3)。
4. **或改走落地框架:** 強調「打敗重現版商用 API、CPU ~120ms、可自架」,把主論文投 **NAACL 2027 Industry Track(直投,10/31 截止)**(與路線 A 擇一)。
5. **Workshop(負面結果/效率/評估):** 2027 清單尚未公布。依 ACL 2027 聯合 workshop call 的**建議時程**,下一批 workshop 論文截止約為:**EACL 2027 workshops ≈ 2026-12-15**、**ICLR 2027 workshops ≈ 2027-02-01**、**NAACL 2027 workshops ≈ 2027-02-05**。清單一出就挑 negative results / efficiency / evaluation / conversational AI 主題的去投(非典藏軌可與主論文並行)。

---

## 🗓️ 投稿時間軸(還開著的) · Live submission calendar (as of 2026-10-04)

| 截止日 / Deadline | Venue / 管道 | 路由 / Route | 性質 / Note |
|---|---|---|---|
| **隨時** / rolling | **arXiv cs.CL** | 直接 | preprint,先做(首投需個人 endorse) |
| **隨時** / rolling | **TMLR** | OpenReview | 期刊,免費,無死線 |
| **隨時** / rolling | **Computational Linguistics / JAIR** | 各自平台 | 期刊,免費,無死線 |
| **2026-10-12**(AoE)【已查證】 | **ARR 2026 Oct cycle** | ARR/OpenReview | → commit NAACL/COLING 2027 |
| **2026-10-31**(AoE)【已查證】 | **NAACL 2027 Industry Track** | 直投 OpenReview | 不走 ARR |
| **2026-11-01** 23:59 HST【已查證】 | **TACL**(之後每月 1 號) | OpenReview | 期刊,免費,雙盲 |
| 2026-11-01 開放 → **2026-12-04**【已查證】 | **NAACL 2027 System Demonstrations** | 直投 OpenReview | 單盲;內容須與主論文區隔 |
| **≈2026-12-15**(建議時程) | **EACL 2027 workshops**(清單未公布;ARR commit ≈12/22) | 各 workshop | 2027 第一批 workshop |
| **2026-12-23**【已查證】 | ARR commitment → **NAACL 2027 / COLING 2027 主會議** | ARR | 須先在 10/12 cycle 內 |
| **2027-01-04**【已查證】 | **ARR 2027 Jan cycle → ACL 2027**(京都 8/17–22) | ARR | 路線 A 的下一班車 |
| **≈2027-02-01**(建議時程) | **ICLR 2027 workshops**(清單未公布;ICBINB 2027 未宣布) | 各 workshop | ICLR 2027:4/26–30,workshop 4/29–30 |
| **≈2027-02-05**(建議時程) | **NAACL 2027 workshops**(清單未公布;ARR commit ≈3/12) | 各 workshop | — |

> 其餘 2026 cycle venue(EMNLP 2026 主/Industry/Demo、AACL 2026、Insights@EMNLP2026、所有 NeurIPS 2026 workshop、ICBINB@ICLR2026…)**均已截止**。

---

## 完整清單 A · 主會議與 ARR 各 track · Main conferences & ARR tracks

所有 ACL 家族主 track 都走 **ACL Rolling Review (ARR)**:投一次到 ARR → 審完 → commit 到某個 venue 由其決定收否。**ARR 無投稿費,ACL Anthology 完全開放、無 APC。** ARR 審稿雙盲。
All ACL-family main tracks go through **ARR**. **No submission fee; ACL Anthology is free OA.** Double-blind.

**ARR 2026 cycle 截止日【已查證】:** Jan 5 / Mar 16 / May 25 / Aug 3 / **Oct 12**(皆 2026)。**ARR 2027 各 cycle 日期【官方未公布】**,僅知 ACL 2027 對應 2027 年 1 月 cycle(1/4)。
【src: https://github.com/acl-org/aclrollingreview/blob/main/dates.md 、 https://aclrollingreview.org/authors 】

| Venue / track | 狀態 / Status | 截止 / Deadline | 地點 / Location | 格式 / Format | 適配 / Fit | 來源 |
|---|---|---|---|---|---|---|
| **NAACL 2027 主會議** ⭐ | **活 / LIVE** | ARR **10/12/2026** → commit **12/23/2026**;通知 2/10/2027;camera-ready 3/3/2027;會議 **6/1–5/2027**(皆 AoE)【已查證】 | San Francisco, USA | 長 8pp / 短 4pp(+1 camera-ready),refs/appendix 不計,須 limitations 段,ACL/ARR 模板,雙盲。**FAQ:錄取可為 Main 或 Findings** | **5/5** — matched-budget + McNemar + 負面結果 + 自我批判,就是主會議實證論文 | https://2027.naacl.org/calls/main_conference_papers/ 、 https://2027.naacl.org/faq/ |
| **COLING 2027 主會議** | **活 / LIVE** | 同 ARR 10/12 cycle → commit 12/23;通知 2/10/2027;**camera-ready【官方未公布】**;會議 **5/9–14/2027**(線上 5/6–7)【已查證】 | Macau, China | 長 8pp / 短 4pp;ACL/ARR 模板。**FAQ:「Acceptances at COLING are in the Main Conference proceedings」→ 無 Findings**。主 call 未另述匿名規則(ARR 階段雙盲) | **4/5** — 與 NAACL 擇一 commit;無 Findings 退路 | https://2027.coling-iccl.org/calls/main_conference_papers/ 、 https://2027.coling-iccl.org/faq/ |
| **ACL 2027 主會議** ⭐ | **下一班 / NEXT** | ARR **2027-01-04**(「at latest, by the ARR 2027 January cycle」);commit/通知/camera-ready【官方 TBA】【已查證】 | **Kyoto, Japan**,8/17–22/2027(主會議 8/20–22) | 長/短;**未入選 main 者自動考慮 Findings** | **5/5** — 錯過 10/12 時的最佳主會議選項 | https://2027.aclweb.org/ 、 https://2027.aclweb.org/calls/main/ |
| **EMNLP 2027** | **未正式公布** | 對應哪個 ARR cycle【官方未公布】 | 「tentatively early November 2027 … Mexico or Central America」(混合制) | — | **5/5(最佳 thematic)** — 盯 ARR 2027 日期公布 | https://www.aclweb.org/portal/content/joint-call-workshops-proposals-2027 |
| ACL 2026 | **已過 / PASSED** | 7/2026 已舉行 | San Diego | 有 Findings | → 改看 ACL 2027 | https://2026.aclweb.org/ |
| EMNLP 2026 | **已過 / PASSED** | ARR 5/25/2026;會議 10/24–29/2026 | Budapest | 有 Findings | → 改看 EMNLP 2027 | https://2026.emnlp.org/ |
| EACL 2027 | **形同錯過** | ARR 8/3/2026;commit 10/11(限 8 月 cycle 稿件);會議 3/9–14/2027 | Athens | ACL/ARR | 新稿無法進 | https://2027.eacl.org/calls/papers/ |
| AACL-IJCNLP 2026 | **已過 / PASSED** | ARR 5/25/2026;會議 11/6–10/2026 | Hengqin | ACL/ARR | 關閉 | https://2026.aaclnet.org/ |

### 費用參考:ACL 家族會議註冊費 · Registration fees (ACL-family reference)

NAACL 2027(「Coming soon!」)與 COLING 2027(「will be announced soon」)**註冊費【官方未公布】**。最新可參考的是 **EMNLP 2026(布達佩斯)實際費率【已查證】**:

| 類別 | 實體 Author/Presenter | 實體 Early | 實體 Late | 實體 Onsite | 線上 Author | 線上 Early | 線上 Late |
|---|---|---|---|---|---|---|---|
| Student | $550 | $350 | $500 | $600 | $250 | $150 | $300 |
| Academic | $850 | $650 | $800 | $900 | $400 | $300 | $450 |
| Industry | $1,150 | $950 | $1,100 | $1,200 | $550 | $450 | $600 |

- **另加每篇論文報告費**:實體 $200(1–2 篇)、第 3–6 篇各 $100;線上 $100 / 各 $50。「Accepted Findings papers that are not being presented are not required to pay the Paper Presentation Registration Fee」。
- **ACL 會員(必須)**:一般 1 年 $100、學生 $50;非高所得國家會員優惠 $50 / 學生 $25。
- 【src: https://2026.emnlp.org/registration/ 、 https://www.aclweb.org/portal/content/membership-fees 】
- → 獨立作者以「academic/industry 實體 + 會員 + 報告費」粗估約 **US$950–1,500**;線上報告可大幅降低。

---

## 完整清單 B · Industry Track 與 System Demonstrations(直投,不走 ARR)· Direct-submission tracks

| Venue / track | 狀態 | 截止 / Deadline | 格式 / Format | 為何契合 / Why it fits | 來源 |
|---|---|---|---|---|---|
| **NAACL 2027 Industry Track** ⭐ | **活 / LIVE** | **10/31/2026**(AoE);通知 2/24/2027;camera-ready【官方 TBD】【已查證】 | **6 pages**(+1);直投 OpenReview,不走 ARR;與主會議擇一 | 落地意圖分類、勝過重現版商用 API、CPU ~120ms、可自架服務——正中 track 要旨(不必 SOTA) | https://2027.naacl.org/calls/industry_track/ |
| **NAACL 2027 System Demonstrations** ⭐ | **活 / LIVE** | 投稿系統 **11/1/2026** 開放 → **12/4/2026** 截止;通知 2/10/2027;camera-ready 3/3/2027【已查證】 | **≤6 pages**(超過即 desk reject),appendix ≤2pp;**單盲(不需匿名)**;須附論文 + ≤2.5 分鐘錄影 + 可用 demo/套件;無評估 = desk reject;**禁止與審查中/已發表論文的內容或結果大幅重疊** | 已有現場 demo + 可自架 MIT 服務 + 已量測;**論文須寫系統,不重寫主論文實驗** | https://2027.naacl.org/calls/system_demonstration/ |
| COLING 2027 Industry | 參考 | camera-ready【官方 TBD】;**2027 起改為雙盲** | — | 備案 | https://2027.coling-iccl.org/calls/industry_track/ |
| EMNLP 2026 Industry / Demo | **已過 / PASSED** | Industry 6/16;Demo 7/10(2026) | 6pp | 關閉 → 看 2027 | https://2026.emnlp.org/calls/industry_track/ |

---

## 完整清單 C · Workshops(主題最契合,但 2026 cycle 全過、2027 清單未出)· Workshops

**現況(2026-10-04 查證):** 最對味的 thematic workshops 多數停辦;EACL / NAACL 2027 的 workshop 清單**都還沒公布**(EACL 2027 workshops 頁 404;NAACL 2027「Coming soon!」)。2027 聯合 workshop 提案已於 9/4 截止、**10/2 放榜**,且「there will not be a second call」,所以清單應很快出爐。

**2027 workshop 論文建議時程(各 workshop 可自訂)【已查證】:**

| | EACL 2027 workshops | NAACL 2027 workshops | ICLR 2027 workshops |
|---|---|---|---|
| 第一次 CFP | 2026-10-13 | 2026-10-26 | (提案 10/9 截止,11/29 放榜) |
| 直投截止 | **2026-12-15** | **2027-02-05** | **≈2027-02-01** |
| ARR commit 截止 | 2026-12-22 | 2027-03-12 | — |
| 通知 | 2027-01-05 | 2027-03-26 | — |

【src: https://2027.eacl.org/calls/workshops/ 、 https://iclr.cc/Conferences/2027/CallForWorkshops 】聯合 call 也寫明「workshops may also accept non-archival submissions, such as findings papers」。

| Workshop | 契合 / Fit | 典藏? / Archival | 2026-10-04 狀態 / Status | 下一步 / Next | 來源 |
|---|---|---|---|---|---|
| **Insights from Negative Results in NLP** | **5/5**(負面結果 + 自我批判正是其 mandate) | **典藏 short ≤4pp(錄取 +1 頁)或 非典藏 1–2pp 摘要(限已在別處發表的論文)** ✅;今年無 long 軌 | 第 7 屆 @ EMNLP 2026(布達佩斯);**投稿 6/8/2026 已過**【已查證】;**2027 未宣布** | 盯 2027 workshop 清單。典藏軌禁與「審查期重疊」的投稿並行 | https://insights-workshop.github.io/2026/cfp |
| **SustaiNLP**(Simple & Efficient NLP) | **5/5** | 典藏(歷年) | **停辦**:最後一屆 ACL 2023,2024–2027 無【已查證】。替代:ACL 2026 曾辦「SELVA」(Sustainable and Efficient Language, Vision, and Action Models,已過) | 盯 2027 清單是否有 SELVA 續辦等效率主題 | https://aclanthology.org/venues/sustainlp/ |
| **ENLSP @ NeurIPS** | **5/5** | 非典藏(歷年) | **停辦**:2024 後無,不在 NeurIPS 2025/2026 清單【已查證】 | 改看 NeurIPS 2027 效率類 workshop | https://neurips2024-enlsp.github.io/ |
| **NLP4ConvAI** | **5/5** | 典藏 + 非典藏可選 | **停辦**:最後一屆 ACL 2024,2025–2027 無【已查證】 | 改看對話類 workshop | https://aclanthology.org/venues/nlp4convai/ |
| **Eval4NLP** | **4/5** | 典藏 或 非典藏 ✅ | **2026 無**(不在 AACL 2026 的 12 個 workshop 中)【已查證】;最後一屆 2025 | 盯 EACL/NAACL 2027 清單;替代:GEM、EvalEval(ACL 系列評估 workshop) | https://2026.aaclnet.org/program/workshops/ |
| **ICBINB**("I Can't Believe It's Not Better") | **4/5** | **永遠有非典藏選項**(引文見規則 4)✅;ICLR 2026 版 ≤4pp | ICLR 2026 版 1/31 已過;**ICLR 2027 版未宣布**(ICLR 2027 workshop 名單 11/29 才放榜);NeurIPS 2026 只有生物版【已查證】 | 11/29 後看 ICLR 2027 workshop 名單 | https://sites.google.com/view/icbinb-2026/submit |
| **RepL4NLP** | 4/5 | 典藏 + 非典藏可選 | 最後一屆 NAACL 2025;2026 無 | 盯 2027 清單 | https://aclanthology.org/venues/repl4nlp/ |
| NeurIPS 2026 效率類(LIGHT / ODI / AXIOM / SLM-Agents / Resource-Aware Agentic AI) | 3–4/5 | 多為非典藏 | NeurIPS 建議論文截止 8/29/2026,**多半已過**(未逐一確認各自頁面) | 看 NeurIPS 2027 | https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/ |
| BlackboxNLP 2026 | 2/5 | 典藏 8pp + 非典藏 2pp | 7/17 已過 | 低優先 | https://blackboxnlp.github.io/2026/ |
| UncertaiNLP 2026 | 2/5 | 典藏/非典藏 | 8/7 已過 | 不建議硬湊(本文未做 UQ) | https://uncertainlp.github.io/2026/ |
| SIGDIAL 2026 | 3/5 | 典藏 | 已過 | 下屆參考 | https://2026.sigdial.org/ |

> **非典藏、對一稿多投最友善:** Insights(1–2pp 摘要軌)、ICBINB、Eval4NLP(Presenting-Published 軌)、BlackboxNLP(2pp)、多數 NeurIPS 效率 workshop。

---

## 完整清單 D · 期刊(獨立作者最友善:免費 + 無死線)· Journals & reproducibility venues

| # | Venue | 截止 / Model | 費用 / Fee | 格式 / Format | 審稿速度 | 契合 | 來源 |
|---|---|---|---|---|---|---|---|
| 1 | **TMLR** ⭐ | **滾動、無死線** | **免費**:「imposes no fees or payments to authors」【已查證】 | 無硬頁數上限;OpenReview;**雙盲**;不收會議論文擴充版 | 無單一官方總目標;各階段目標:AE 指派 1 週內、審稿 2 週(≤12 頁)/4 週(>12 頁)、AE 於討論開始後 5 週內提決定 → **加總約 2–3 個月**(此為推算) | **5/5** — 收錄只看兩條(證據充分 + 有人感興趣),**novelty 非必要**。⚠️ **Reproducibility Certification 多半不適用**:它頒給「primary purpose is reproduction of other published work」的論文,Quorum 主體是原創集成方法 | https://jmlr.org/tmlr/editorial-policies.html 、 https://jmlr.org/tmlr/acceptance-criteria.html |
| 2 | **TACL** ⭐ | **每月 1 號 23:59 夏威夷時間**;下個 **2026-11-01**【已查證】 | **免費**:「imposes neither author processing charges nor submission charges」【已查證】 | **正文 10 頁**(refs 不計);2024-03 起 appendix 不計入,重現細節 ≤5 頁、補充結果 ≤3 頁;TACL 模板(A4);**雙盲**,禁止洩漏身分的自我引用 | ~6 週目標(歷年約 46–60 天,為社群數據) | **5/5** — correctness 導向 NLP 期刊;錄取後可選在 ACL/NAACL/EACL/EMNLP/AACL 口頭報告(非必要) | https://transacl.org/index.php/tacl/about/submissions 、 https://transacl.org/index.php/tacl/announcement/view/105 |
| 3 | **Computational Linguistics (CL)** | **滾動**(「accepted on a rolling basis」)【已查證】 | **免費**:「does not charge any submission, processing or publication fees」【已查證】 | Long ≤40pp / Short ≤20pp / Squib ≤8pp;**單盲** | 目標審稿:≤20 頁 3 週、≤40 頁 4 週、≤45 頁 5 週、更長 6 週(不保證);**實際平均首次決定 16.5 天(排除 desk reject 為 55 天)**;2025 IF 13.4【已查證】 | **4/5** — Short 軌明文「Well-analyzed negative results … are also suitable」 | https://submissions.cljournal.org/index.php/cljournal/about 、 https://submissions.cljournal.org/index.php/cljournal |
| 4 | **JAIR** | 滾動 | **免費**(CC BY) | 無硬頁數;**須 LaTeX**(不符即退);單盲 | ~8–12 週 | 3/5 — 免費/可敬,但無 novelty-agnostic 專軌 | https://www.jair.org/index.php/jair/about/submissions |
| 5 | **PeerJ Computer Science** | 滾動 | **貴:APC US$2,155**(未稅)【已查證】;先前提到的 $695 / $1,395 **皆為過時資訊**。另有終身會員方案(Basic $755 起,但**所有作者都須是會員**) | 無硬字數(>35 頁加收);單盲 | 首次決定 30–35 天;接受到出版 32 天;接受率 33%【已查證】 | 4/5 範圍/速度;**費用是主要障礙** | https://peerj.com/pricing/ 、 https://peerj.com/journals/computer-science/ |
| 6 | **Natural Language Processing**(前 JNLE,Cambridge) | 隨時經 ScholarOne 投稿(官方未使用「rolling」字樣);年出 6 期 + FirstView | **最貴:APC £2,610 / US$3,655**(可能另加稅)【已查證】 | 單一匿名(single-anonymized),≥2 位外審;Cambridge 模板 | **【官方未公布】**(metrics 頁:「will soon be providing annual statistics」) | 3/5 — 落地範圍契合但 APC 最高 | https://www.cambridge.org/core/journals/natural-language-processing/information/author-instructions/fees-and-pricing |
| 7 | **arXiv (cs.CL)** ⭐ | 滾動 | **免費** | 6 種授權擇一且**不可更改**:CC BY 4.0(建議,呼應 MIT)、CC BY-SA 4.0、CC BY-NC-SA 4.0、CC BY-NC-ND 4.0、arXiv non-exclusive license 1.0、CC0【已查證】 | 次工作日上架 | **5/5 作為「先行步驟」** | https://info.arxiv.org/help/license/index.html |

**APC 減免條件【已查證】(獨立作者必看):**

- **PeerJ CS**:世界銀行「低所得經濟體」國家作者可申請「no questions asked」減免(每人每年一次、須在審稿前核准);大學部學生共同作者有會員減免;機構會員(AIM)可能全額支付。**一般的經濟困難減免【官方未載明】**;申請管道在現行頁面上【官方未載明】。【src: https://peerj.com/questions/6089-are-any-waivers-or-discounts-available-for-lower 】
- **Cambridge NLP**:Cambridge Open Equity Initiative 自動為 100+ 個中低所得國家的通訊作者付 Gold OA;其餘作者可在**錄取後**填 Author Information Form 時申請折扣或全免(不保證)。【src: https://www.cambridge.org/core/journals/natural-language-processing/information/journal-policies/open-access-options 】

**arXiv endorsement(2026 年新規)【已查證】:** 自 **2026-01-21** 起,學校信箱**不再能單獨**作為新作者的 endorsement 條件;自動 endorsement 須同時具備「機構信箱」與「已有同一 endorsement domain 的 arXiv 論文」。否則須找**同領域的現役 arXiv 作者個人 endorse**,且「arXiv staff cannot waive endorsement requirements」。→ 獨立作者首投 cs.CL:**一定要先找 endorser**。【src: https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/ 、 https://info.arxiv.org/help/endorsement.html 】

> **ReScience C:不適用(1/5)【已查證,修正前版】。** 官方 FAQ:「Can I submit the replication of my own research? **No.**」;作者指南:「you cannot submit the replication of your own research」;且被重現的原研究須是「already published research」。Quorum 是作者自己的方法,被比較的對象(社群對閉源 API 的重現)也不是已發表論文 → 不符。(免費、收負面結果這兩點屬實,但範圍不合。)【src: https://rescience.github.io/faq/ 、 https://rescience.github.io/write/ 】
> **MLRC(NeurIPS 2026 可重現性軌):不適用(1/5)。** 作業單位是「重現**別人**已發表的論文」,且 TMLR-decision 死線 9/30/2026 已過。【src: https://reproml.org/ 】
> **JASNH / 心理學 null-result 期刊:領域不符。**

---

## 完整清單 E · 已過期 / 停辦 / 不建議 · Passed / dormant / not recommended

- **已過 2026 cycle:** ACL 2026、EMNLP 2026(主/Industry/Demo)、AACL 2026、Insights@EMNLP2026、ICBINB@ICLR2026、NeurIPS 2026 各 workshop、BlackboxNLP/UncertaiNLP 2026、NeurIPS 2026 Evaluations & Datasets、SIGDIAL 2026。
- **停辦(2026-10-04 查證仍無新版):** SustaiNLP(最後 2023)、ENLSP(最後 2024)、NLP4ConvAI(最後 2024)、Eval4NLP(2026 無)、RepL4NLP(最後 2025)。
- **不建議:** **ReScience C**(不收自己研究的重現)、MLRC(重現別人論文用)、UncertaiNLP / EIML(本文未做 UQ)、JASNH(領域不符)。

---

## 查證紀錄(原「待查證清單」逐項結案)· Verification log — former to-verify list, closed item by item

| # | 原待查證項目 | 查證結果(2026-10-04) | 狀態 |
|---|---|---|---|
| 1 | TACL 下一個確切截止日 | **2026-11-01 23:59 夏威夷時間**,之後每月 1 號;正文 10 頁,appendix 另計(≤5 / ≤3 頁);免費 | ✅ 已查證 |
| 2a | NAACL 2027 是否有 Findings | 主 call 未提;**FAQ:錄取可為 Main 或 Findings** | ✅ 已查證 |
| 2b | COLING 2027 是否有 Findings | **沒有**:FAQ「Acceptances at COLING are in the Main Conference proceedings」 | ✅ 已查證 |
| 2c | NAACL / COLING 2027 註冊費 | **官方尚未公布**(NAACL「Coming soon!」、COLING「announced soon」);改附 EMNLP 2026 實際費率 + ACL 會費作參考 | ✅ 已結案(官方未公布) |
| 2d | COLING 2027 camera-ready 日 | **官方未公布**(主 call 與首頁皆無;industry 軌標 TBD) | ✅ 已結案(官方未公布) |
| 2e | COLING 2027 是否雙盲 | 主 call 未另述(論文經 ARR,ARR 為雙盲);COLING Industry 軌 2027 起改雙盲 | ✅ 已結案 |
| 3 | EMNLP 2027 cycle/日期 | 暫定 **2027 年 11 月初,墨西哥或中美洲**,混合制;對應 ARR cycle 與 ARR 2027 日期**官方未公布**。順帶發現 **ACL 2027 已公布**(京都 8/17–22,ARR 2027-01-04) | ✅ 已結案 |
| 4a | PeerJ CS APC 與減免 | **US$2,155**($695/$1,395 皆過時);低所得國家免審減免、大學部學生、機構會員;一般困難減免官方未載明 | ✅ 已查證 |
| 4b | Cambridge NLP 時程 | **官方未公布**審稿時程;APC £2,610/$3,655;COEI 與錄取後申請減免 | ✅ 已結案(官方未公布) |
| 5a | ICBINB @ ICLR 2027 截止與典藏 | **ICLR 2027 版尚未宣布**(workshop 名單 11/29 放榜;建議論文截止 ≈2027-02-01);典藏:**永遠有非典藏選項** | ✅ 已結案 |
| 5b | SustaiNLP / ENLSP / NLP4ConvAI / Eval4NLP 是否復辦 | 四者 **2026–2027 皆無新版**;EACL/NAACL 2027 workshop 清單尚未公布(提案 10/2 放榜) | ✅ 已查證 |
| 6 | 註冊費為 2025 基準估值 | 已換成 **EMNLP 2026 官方實際費率**(含報告費與 ACL 會費) | ✅ 已查證 |
| + | 查證時發現的前版錯誤 | ① **ReScience C 不收自己研究的重現 → 改為不適用**;② TMLR Reproducibility Cert. 多半不適用;③ Demo 與主論文並行**須內容區隔**(NAACL Demo 禁大幅重疊、TACL 禁任何材料同時審查);④ arXiv 2026-01-21 endorsement 新規;⑤ PeerJ 舊價格過時 | ✅ 已修正 |

> **官方未公布的項目無法再查證**,只能等主辦方公布:NAACL/COLING 2027 註冊費、COLING 2027 camera-ready、EMNLP 2027 對應 cycle、ARR 2027 日期、2027 各 workshop 清單、Cambridge NLP 審稿時程。投稿前請再看一次官方頁。

---

## 一頁總結:值得投的所有管道 · One-page shortlist of everything worth submitting to

| 優先 | Venue | 現在能投? | 費用 | 契合 | 角色 |
|---|---|---|---|---|---|
| ★★★ | **arXiv cs.CL** | ✅ 隨時(先找 endorser) | 免費 | 5/5 | preprint,**先做** |
| ★★★ | **ARR Oct → NAACL 2027 主/Findings** | ✅ 10/12 | 免費投稿 | 5/5 | 典藏主家(路線 A) |
| ★★★ | **TMLR** | ✅ 隨時 | 免費 | 5/5 | 典藏主家(路線 B,期刊) |
| ★★★ | **TACL** | ✅ 11/1 起每月 1 號 | 免費 | 5/5 | 典藏主家(路線 B,NLP 原生) |
| ★★★ | **NAACL 2027 System Demo** | ✅ 11/1 開放、12/4 截止 | 免費投稿 | 5/5 | 條件式並行(內容須與主論文區隔) |
| ★★☆ | **ARR Jan 2027 → ACL 2027** | ⏳ 2027-01-04 | 免費投稿 | 5/5 | 錯過 10/12 時的路線 A |
| ★★☆ | **NAACL 2027 Industry Track** | ✅ 10/31 | 免費投稿 | 5/5 | 落地框架(與主會議擇一) |
| ★★☆ | **COLING 2027 主** | ✅ 同 ARR 10/12 | 免費投稿 | 4/5 | 與 NAACL commit 擇一(無 Findings) |
| ★★☆ | **Computational Linguistics** | ✅ 隨時 | 免費 | 4/5 | 免費期刊備案(歡迎負面結果、首次決定快) |
| ★★☆ | **EACL / ICLR / NAACL 2027 workshops** | ⏳ ≈12/15、≈2/1、≈2/5 | 註冊費 | 4–5/5(視清單) | 負面結果/效率/評估(非典藏可並行) |
| ★☆☆ | JAIR | ✅ 隨時 | 免費 | 3/5 | 期刊備案 |
| ★☆☆ | PeerJ CS / Cambridge NLP | ✅ 隨時 | **$2,155 / $3,655** | 3–4/5 | 僅在有經費或符合減免時考慮 |
| ✗ | ReScience C / MLRC | — | — | 1/5 | **不適用**(只收重現他人已發表研究) |

---

*本頁由三個背景研究 session 的結果彙整、去重、交叉校正而成,並於同日對原「待查證清單」逐項對照官方頁面二次查證(2026-10-04)。發現過時請回報,我會更新。*
*Compiled from three background research runs and re-verified item by item against official pages on 2026-10-04. Re-check each official CFP before submitting.*
