import { DocumentPageResponse, SourceSelector } from "@/types/api";

export interface HighlightRange {
  start: number;
  end: number;
  selectorIndex: number;
}

export interface HighlightResolution {
  ranges: HighlightRange[];
  warning: string | null;
}

export function resolveHighlightRanges(
  page: DocumentPageResponse,
  selectors: SourceSelector[],
): HighlightResolution {
  if (selectors.length === 0) return { ranges: [], warning: null };
  const relevant = selectors.filter(
    (selector) => selector.page == null || selector.page === page.page_number,
  );
  if (relevant.length === 0) {
    return { ranges: [], warning: "Bu səhifə üçün evidence span yoxdur." };
  }
  const ranges: HighlightRange[] = [];
  for (const [selectorIndex, selector] of relevant.entries()) {
    if (
      selector.representation_id !== page.representation_id ||
      selector.source_sha256 !== page.source_sha256
    ) {
      return { ranges: [], warning: "Sənəd versiyası evidence ilə uyğun deyil." };
    }
    const exact = selector.quote.exact;
    const pagePosition = selector.page_position;
    if (
      pagePosition &&
      page.page_content.slice(pagePosition.start, pagePosition.end) === exact
    ) {
      ranges.push({ ...pagePosition, selectorIndex });
      continue;
    }
    const matches = quoteMatches(
      page.page_content,
      exact,
      selector.quote.prefix,
      selector.quote.suffix,
    );
    if (matches.length !== 1) {
      return {
        ranges: [],
        warning:
          matches.length === 0
            ? "Exact evidence mətni səhifədə tapılmadı."
            : "Evidence mətni səhifədə unikal deyil; highlight edilmədi.",
      };
    }
    ranges.push({ ...matches[0], selectorIndex });
  }
  ranges.sort((left, right) => left.start - right.start || left.end - right.end);
  for (let index = 1; index < ranges.length; index += 1) {
    if (ranges[index].start < ranges[index - 1].end) {
      return { ranges: [], warning: "Evidence span-ları üst-üstə düşür." };
    }
  }
  return { ranges, warning: null };
}

function quoteMatches(
  text: string,
  exact: string,
  prefix: string,
  suffix: string,
): Array<{ start: number; end: number }> {
  const matches: Array<{ start: number; end: number }> = [];
  let offset = 0;
  while (offset <= text.length) {
    const start = text.indexOf(exact, offset);
    if (start < 0) break;
    const end = start + exact.length;
    const prefixMatches =
      !prefix || text.slice(Math.max(0, start - prefix.length), start) === prefix;
    const suffixMatches = !suffix || text.slice(end, end + suffix.length) === suffix;
    if (prefixMatches && suffixMatches) matches.push({ start, end });
    offset = start + 1;
  }
  return matches;
}

