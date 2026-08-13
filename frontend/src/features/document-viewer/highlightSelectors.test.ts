import { describe, expect, it } from "vitest";

import { DocumentPageResponse, SourceSelector } from "../../types/api";
import { resolveHighlightRanges } from "./highlightSelectors";

const page: DocumentPageResponse = {
  document_id: "doc-1",
  document_version_id: "dver-1",
  page_number: 1,
  page_content: "Maddə 1. Birinci sübut. İkinci sübut.",
  article_information: [],
  representation_id: "cleaned:dver-1:hash",
  source_sha256: "hash",
  page_start_offset: 20,
};

function selector(exact: string, start: number, end: number): SourceSelector {
  return {
    representation_id: "cleaned:dver-1:hash",
    representation_type: "cleaned_markdown",
    document_version_id: "dver-1",
    source_sha256: "hash",
    page: 1,
    position: { start: start + 20, end: end + 20 },
    page_position: { start, end },
    quote: { exact, prefix: "", suffix: "" },
    normalization_version: "cleaned_markdown_utf8_codepoint_v1",
  };
}

describe("resolveHighlightRanges", () => {
  it("resolves multiple exact page positions", () => {
    const first = page.page_content.indexOf("Birinci sübut");
    const second = page.page_content.indexOf("İkinci sübut");
    const result = resolveHighlightRanges(page, [
      selector("Birinci sübut", first, first + "Birinci sübut".length),
      selector("İkinci sübut", second, second + "İkinci sübut".length),
    ]);
    expect(result.warning).toBeNull();
    expect(result.ranges).toHaveLength(2);
  });

  it("uses unique exact quote when position changed", () => {
    const target = page.page_content.indexOf("İkinci sübut");
    const value = selector("İkinci sübut", 0, 3);
    const result = resolveHighlightRanges(page, [value]);
    expect(result.warning).toBeNull();
    expect(result.ranges[0]).toMatchObject({
      start: target,
      end: target + "İkinci sübut".length,
    });
  });

  it("refuses highlight when source hash differs", () => {
    const value = selector("Birinci sübut", 9, 22);
    value.source_sha256 = "different";
    const result = resolveHighlightRanges(page, [value]);
    expect(result.ranges).toEqual([]);
    expect(result.warning).toContain("uyğun deyil");
  });
});

