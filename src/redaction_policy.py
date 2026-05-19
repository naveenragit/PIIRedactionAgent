"""Custom redaction rules used by both the redactor and reviewer agents."""

CUSTOM_REDACTION_RULES = """
Purpose
This is a compliance task. The goal is to take a banker's pitch deck (PDF
or PPTX exported to PDF) that contains client information and produce a
sanitized copy where personally identifiable information (PII) and material
non-public information (MNPI) are covered by black rectangles, while the
overall structure and essence of the content are preserved.

General guidelines
- Maintain the original layout and design as much as possible.
- Do not rewrite sentences or obscure the banker's judgment unnecessarily.
- Cover all sensitive information described below.
- Use consistent replacements for sensitive terms across every page.

Categories to cover (every constituent word of each span should be marked)
1. Client identity (cover on every page, every occurrence)
   - Client's full legal name, brand name, common abbreviations, and stock
     ticker symbols. Conceptually replaced with "the client" or
     "Pharma Company".
   - Subsidiary, division, and brand-specific terms tied to the client.
   - Any descriptor that uniquely identifies the client (HQ city + sector
     combinations, distinctive product names).
2. People and contacts
   - Names of executives, board members, advisors, employees (CEO, CFO,
     etc.). Conceptually replaced with "XXX" or "Name".
   - Direct contact info: phone, email, physical address.
3. Valuation, revenue, and financial metrics
   - Specific dollar amounts, revenue, EBITDA, margins, valuation multiples,
     share prices, market caps, growth rates expressed as concrete numbers,
     deal sizes. Conceptually replaced with "X.X".
   - Cover the numeric tokens together with adjacent currency / units that
     disclose the figure (for example "$", "4.2", "B" should be covered
     together).
4. Identifying events and MNPI
   - References to recent or pending transactions, M&A activity, product
     launches, regulatory milestones, or any non-public event that could
     identify the client or constitute MNPI.
   - Precedent-deal target names, transaction dates, and value labels in
     charts and tables.
5. Standard PII identifiers
   - SSN/EIN, passport, driver's license, DOB, bank/credit card/IBAN/routing
     numbers, government IDs.
6. Client logos, brand marks, and watermark imagery
   - Every visible company logo, brand mark, or watermark that identifies
     the client must be covered, whether it appears as a small inline
     graphic (header, footer, slide corner) or as a large background
     watermark.
   - Use `list_visual_regions` to discover them and `redact_visual_regions`
     to mark them. Trust the `strategy` field returned by the tool:
     * `inline` regions are covered by a black rectangle in place.
     * `page_split` regions are background watermarks; redacting them
       would also obscure overlaid foreground content, so the renderer
       automatically emits the fully blacked page followed by a clean
       `Page N (continued)` reflow page containing only the non-redacted
       text. Treat this as the expected behavior; do not work around it.
   - The reviewer must call `detect_logos_on_rendered_pdf` and flag every
     remaining graphic as `type: "Logo"`.

What to leave intact
- Generic industry commentary, market context, methodology, framework names.
- Banker's qualitative judgment, headings, section titles that don't name
  the client, page numbers, and standard boilerplate disclaimers.
- Bullet structure and ordering — do not reflow content.

Consistency
- Every occurrence of the same client name, ticker, or executive should be
  covered on every page.
- Every disclosed dollar or percentage figure tied to the client should be
  covered, even if it appears in a chart label or footnote.
- Use the `redact_all_matching_terms` tool to sweep client-identity terms
  document-wide rather than relying on per-page enumeration.
"""
