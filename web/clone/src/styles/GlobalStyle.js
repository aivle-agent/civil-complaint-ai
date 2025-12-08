import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  :root {
    /* Primary Colors */
    --primary5: #ecf2fe;
    --primary10: #d8e5fd;
    --primary20: #b1cefb;
    --primary30: #86aff9;
    --primary40: #4c87f6;
    --primary50: #256ef4;
    --primary60: #0b50d0;
    --primary70: #083891;
    --primary80: #052561;
    --primary90: #03163a;
    --primary95: #020f27;

    /* Secondary Colors */
    --secondary5: #eef2f7;
    --secondary10: #d6e0eb;
    --secondary20: #bacbde;
    --secondary30: #90b0d5;
    --secondary40: #6b96c7;
    --secondary50: #346fb2;
    --secondary60: #1c589c;
    --secondary70: #063a74;
    --secondary80: #052b57;
    --secondary90: #031f3f;
    --secondary95: #02162c;

    /* Grayscale */
    --gray0: #ffffff;
    --gray5: #f4f5f6;
    --gray10: #e6e8ea;
    --gray20: #cdd1d5;
    --gray30: #b1b8be;
    --gray40: #8a949e;
    --gray50: #6d7882;
    --gray60: #58616a;
    --gray70: #464c53;
    --gray80: #33363d;
    --gray90: #1e2124;
    --gray95: #131416;
    --gray100: #000000;

    /* Status Colors */
    --danger5: #fdefec;
    --danger40: #f05f42;
    --information5: #e7f4fe;
    --information40: #2098f3;
    --warning5: #fff3db;
    --warning40: #c78500;
    --success5: #eaf6ec;
    --success40: #3fa654;

    /* Spacing & Sizing */
    --contents-padding-x: 24px;
    --radius-xs: 4px;
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 10px;

    /* Typography */
    --pc-fz-heading-lg: 3.2rem;
    --pc-fz-heading-md: 2.4rem;
    --pc-fz-heading-sm: 1.9rem;
    --pc-fz-heading-xs: 1.7rem;
  }

  /* Reset & Base Styles */
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  html {
    font-size: 62.5%; /* 1rem = 10px */
  }

  body {
    font-family: "Pretendard GOV", "Noto Sans KR", "Inter", sans-serif;
    font-size: 1.6rem;
    color: var(--gray80);
    line-height: 1.5;
    background-color: var(--gray0);
    word-break: keep-all;
  }

  a {
    text-decoration: none;
    color: inherit;
    cursor: pointer;
  }

  ul, ol {
    list-style: none;
  }

  button {
    border: none;
    background: none;
    cursor: pointer;
    font-family: inherit;
  }

  input, select, textarea {
    font-family: inherit;
    font-size: inherit;
  }

  /* Utility Classes */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
  }

  .hide {
    display: none !important;
  }
`;

export default GlobalStyle;
