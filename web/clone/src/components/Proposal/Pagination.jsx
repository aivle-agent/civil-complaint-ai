import React from 'react';
import styled from 'styled-components';

const PaginationWrapper = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: 30px;
  gap: 5px;
`;

const PageButton = styled.button`
  width: 32px;
  height: 32px;
  border: 1px solid var(--gray30);
  background-color: white;
  color: var(--gray60);
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 4px;
  font-size: 1.4rem;

  &:hover {
    background-color: var(--gray5);
    color: var(--primary50);
  }

  &.active {
    background-color: var(--primary50);
    color: white;
    border-color: var(--primary50);
    font-weight: 700;
  }

  &.control {
    background-color: var(--gray5);
  }
`;

const Pagination = () => {
    return (
        <PaginationWrapper>
            <PageButton className="control">«</PageButton>
            <PageButton className="control">‹</PageButton>
            <PageButton className="active">1</PageButton>
            <PageButton>2</PageButton>
            <PageButton>3</PageButton>
            <PageButton>4</PageButton>
            <PageButton>5</PageButton>
            <PageButton>6</PageButton>
            <PageButton>7</PageButton>
            <PageButton>8</PageButton>
            <PageButton>9</PageButton>
            <PageButton>10</PageButton>
            <PageButton className="control">›</PageButton>
            <PageButton className="control">»</PageButton>
        </PaginationWrapper>
    );
};

export default Pagination;
