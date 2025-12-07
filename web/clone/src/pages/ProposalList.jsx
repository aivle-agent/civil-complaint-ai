import React from 'react';
import styled from 'styled-components';
import SearchFilter from '../components/Proposal/SearchFilter';
import DataTable from '../components/Proposal/DataTable';
import Pagination from '../components/Proposal/Pagination';

const PageTitle = styled.h3`
  font-size: 2.4rem;
  font-weight: 700;
  margin-bottom: 20px;
  color: var(--gray90);
`;

const ProposalList = () => {
    return (
        <div>
            <PageTitle>공개제안</PageTitle>
            <SearchFilter />
            <DataTable />
            <Pagination />
        </div>
    );
};

export default ProposalList;
