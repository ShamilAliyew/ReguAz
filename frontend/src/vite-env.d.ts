/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_USE_MOCK_API: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "tailwind-merge" {
  export function twMerge(...classLists: Array<string | number | boolean | undefined | null | any>): string;
}
