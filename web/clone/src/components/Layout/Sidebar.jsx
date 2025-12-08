import React from 'react';
import styled from 'styled-components';

const SidebarWrapper = styled.aside`
  width: 240px;
  flex-shrink: 0;
  margin-right: 40px;
`;

const SidebarTitle = styled.h2`
  font-size: 2.8rem;
  font-weight: 700;
  margin-bottom: 20px;
  color: var(--gray90);
  border-bottom: 2px solid var(--gray90);
  padding-bottom: 15px;
`;

const LNBList = styled.ul`
  border: 1px solid var(--gray20);
  border-top: none;
`;

const LNBItem = styled.li`
  border-bottom: 1px solid var(--gray20);

  a {
    display: block;
    padding: 15px 20px;
    font-size: 1.6rem;
    color: var(--gray70);
    background-color: var(--gray0);
    transition: all 0.2s;
    cursor: pointer;

    &:hover {
      background-color: var(--gray5);
      color: var(--primary50);
      font-weight: 500;
    }

    &.active {
      background-color: var(--primary50);
      color: white;
      font-weight: 700;
    }
  }
`;

const Sidebar = ({ activePage, onNavigate }) => {
  return (
    <SidebarWrapper>
      <SidebarTitle>국민제안</SidebarTitle>
      <LNBList>
        <LNBItem>
          <a
            className={activePage === 'form' ? 'active' : ''}
            onClick={() => onNavigate('form')}
          >
            일반제안
          </a>
        </LNBItem>
        <LNBItem>
          <a
            className={activePage === 'list' ? 'active' : ''}
            onClick={() => onNavigate('list')}
          >
            공개제안
          </a>
        </LNBItem>
        <LNBItem><a href="#">실시제안</a></LNBItem>
        <LNBItem><a href="#">국민제안 소개</a></LNBItem>
        <LNBItem><a href="#">공무원제안</a></LNBItem>
      </LNBList>
    </SidebarWrapper>
  );
};

export default Sidebar;
