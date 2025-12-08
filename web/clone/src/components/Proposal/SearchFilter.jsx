import React from 'react';
import styled from 'styled-components';

const SearchWrapper = styled.div`
  background-color: var(--gray5);
  border: 1px solid var(--gray20);
  padding: 20px;
  margin-bottom: 30px;
  border-radius: var(--radius-sm);
`;

const SearchRow = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 10px;

  &:last-child {
    margin-bottom: 0;
  }
`;

const SearchLabel = styled.div`
  width: 100px;
  font-weight: 700;
  color: var(--gray90);
`;

const SearchInputGroup = styled.div`
  display: flex;
  align-items: center;
  flex: 1;
  gap: 10px;
`;

const Input = styled.input`
  height: 36px;
  border: 1px solid var(--gray30);
  padding: 0 10px;
  border-radius: var(--radius-xs);
  width: 150px;

  &:focus {
    border-color: var(--primary50);
    outline: none;
  }
`;

const Select = styled.select`
  height: 36px;
  border: 1px solid var(--gray30);
  padding: 0 10px;
  border-radius: var(--radius-xs);
  min-width: 120px;

  &:focus {
    border-color: var(--primary50);
    outline: none;
  }
`;

const CheckboxGroup = styled.div`
  display: flex;
  gap: 15px;
  align-items: center;

  label {
    display: flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 20px;
  border-top: 1px solid var(--gray20);
  padding-top: 20px;
`;

const Button = styled.button`
  height: 40px;
  padding: 0 30px;
  border-radius: var(--radius-sm);
  font-weight: 500;
  transition: all 0.2s;

  &.primary {
    background-color: var(--gray90);
    color: white;

    &:hover {
      background-color: black;
    }
  }

  &.secondary {
    background-color: white;
    border: 1px solid var(--gray30);
    color: var(--gray80);

    &:hover {
      background-color: var(--gray5);
    }
  }
`;

const SearchFilter = () => {
    return (
        <SearchWrapper>
            <SearchRow>
                <SearchLabel>기간</SearchLabel>
                <SearchInputGroup>
                    <Select>
                        <option>신청일</option>
                    </Select>
                    <Input type="date" defaultValue="2024-11-03" />
                    <span>~</span>
                    <Input type="date" defaultValue="2025-12-03" />
                    <Select>
                        <option>최신순</option>
                        <option>조회순</option>
                    </Select>
                </SearchInputGroup>
            </SearchRow>
            <SearchRow>
                <SearchLabel>검색어</SearchLabel>
                <SearchInputGroup>
                    <Select>
                        <option>제목</option>
                        <option>내용</option>
                    </Select>
                    <Input type="text" placeholder="검색어를 입력하세요" style={{ width: '300px' }} />
                </SearchInputGroup>
            </SearchRow>
            <SearchRow>
                <SearchLabel>종류</SearchLabel>
                <CheckboxGroup>
                    <label><input type="checkbox" /> 자체포상</label>
                    <label><input type="checkbox" /> 중앙포상</label>
                    <label><input type="checkbox" /> 공동제안</label>
                    <label><input type="checkbox" /> 단체제안</label>
                </CheckboxGroup>
            </SearchRow>
            <ButtonGroup>
                <Button className="primary">검색</Button>
                <Button className="secondary">초기화</Button>
            </ButtonGroup>
        </SearchWrapper>
    );
};

export default SearchFilter;
