import React from 'react';
import styled from 'styled-components';
import GlobalStyle from './styles/GlobalStyle';
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import Footer from './components/Layout/Footer';
import ProposalList from './pages/ProposalList';

const MainContainer = styled.div`
  max-width: 1200px;
  margin: 40px auto;
  padding: 0 24px;
  display: flex;
  min-height: 600px;
`;

const ContentWrapper = styled.main`
  flex: 1;
`;

function App() {
    return (
        <>
            <GlobalStyle />
            <Header />
            <MainContainer>
                <Sidebar />
                <ContentWrapper>
                    <ProposalList />
                </ContentWrapper>
            </MainContainer>
            <Footer />
        </>
    );
}

export default App;
