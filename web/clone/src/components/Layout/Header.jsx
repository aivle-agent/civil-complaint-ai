import React from 'react';
import styled from 'styled-components';

const HeaderWrapper = styled.header`
  width: 100%;
  border-bottom: 1px solid var(--gray20);
  background-color: var(--gray0);
`;

const HeaderTop = styled.div`
  background-color: var(--gray0);
  border-bottom: 1px solid var(--gray10);
`;

const HeaderContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 80px;
`;

const Logo = styled.h1`
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--primary50);
`;

const UtilityArea = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
`;

const SearchBox = styled.div`
  display: flex;
  align-items: center;
  border: 1px solid var(--gray30);
  border-radius: 20px;
  padding: 5px 15px;
  width: 250px;

  input {
    border: none;
    outline: none;
    width: 100%;
    font-size: 1.4rem;
  }

  button {
    color: var(--gray50);
  }
`;

const GNB = styled.nav`
  background-color: var(--primary50);
  color: white;
`;

const GNBList = styled.ul`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  height: 50px;
  align-items: center;
`;

const GNBItem = styled.li`
  margin-right: 40px;
  font-size: 1.8rem;
  font-weight: 500;
  cursor: pointer;
  position: relative;
  height: 100%;
  display: flex;
  align-items: center;

  &.active {
    font-weight: 700;
    border-bottom: 3px solid white;
  }

  &:hover {
    opacity: 0.9;
  }
`;

const Header = () => {
    return (
        <HeaderWrapper>
            <HeaderTop>
                <HeaderContainer>
                    <Logo>국민신문고</Logo>
                    <UtilityArea>
                        <button>화면크기</button>
                        <button>For Foreigners</button>
                        <SearchBox>
                            <input type="text" placeholder="검색어를 입력해주세요" />
                            <button>🔍</button>
                        </SearchBox>
                        <button>로그인</button>
                        <button>회원가입</button>
                        <button>전체메뉴</button>
                    </UtilityArea>
                </HeaderContainer>
            </HeaderTop>
            <GNB>
                <GNBList>
                    <GNBItem>민원</GNBItem>
                    <GNBItem class="active">국민제안</GNBItem>
                    <GNBItem>예산낭비절감</GNBItem>
                    <GNBItem>국민생각함</GNBItem>
                    <GNBItem>정책실현</GNBItem>
                    <GNBItem>고객센터</GNBItem>
                </GNBList>
            </GNB>
        </HeaderWrapper>
    );
};

export default Header;
