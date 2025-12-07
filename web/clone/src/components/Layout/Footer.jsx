import React from 'react';
import styled from 'styled-components';

const FooterWrapper = styled.footer`
  background-color: var(--gray5);
  border-top: 1px solid var(--gray20);
  padding: 40px 0;
  margin-top: 60px;
`;

const FooterContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
`;

const FooterLinks = styled.ul`
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  font-size: 1.4rem;
  color: var(--gray60);

  li a.point {
    font-weight: 700;
    color: var(--primary60);
  }
`;

const AddressInfo = styled.div`
  font-size: 1.4rem;
  color: var(--gray50);
  line-height: 1.6;

  p {
    margin-bottom: 5px;
  }
`;

const Copyright = styled.p`
  margin-top: 15px;
  font-size: 1.3rem;
  color: var(--gray40);
`;

const Footer = () => {
    return (
        <FooterWrapper>
            <FooterContainer>
                <FooterLinks>
                    <li><a href="#" className="point">개인정보처리방침</a></li>
                    <li><a href="#">이용약관</a></li>
                    <li><a href="#">자주하는 질문</a></li>
                    <li><a href="#">오류신고·이용문의</a></li>
                </FooterLinks>
                <AddressInfo>
                    <p>(30102) 세종특별자치시 도움5로 20 정부세종청사 국민권익위원회</p>
                    <p>정부민원안내콜센터 국번없이 110 (무료)</p>
                    <p>국민신문고 시스템 이용관련 문의 1600-8172 (유료)</p>
                </AddressInfo>
                <Copyright>© Anti-Corruption & Civil Rights Commission. All rights reserved.</Copyright>
            </FooterContainer>
        </FooterWrapper>
    );
};

export default Footer;
