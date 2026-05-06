# FE Plan - CV Review System

## Correction

**Update Position modal** does NOT contain `minimum fit score`.

Minimum fit score should be editable directly in Position table: -
Display current score - Small pen/edit icon next to score - Click icon
opens inline edit or mini popover/modal to update score only

------------------------------------------------------------------------

## 1. Tech Stack

-   React + JavaScript
-   Vite
-   React Router DOM
-   TanStack Query
-   Axios
-   Ant Design
-   dayjs
-   react-markdown

------------------------------------------------------------------------

## 2. Folder Structure

``` bash
src/
├── api/
├── pages/
│   ├── hr/
│   └── candidate/
├── components/
│   ├── common/
│   ├── modals/
│   ├── tables/
│   ├── chatbot/
│   └── upload/
├── layouts/
├── hooks/
├── routes/
├── types/
└── utils/
```

------------------------------------------------------------------------

## 3. Routing

### HR

``` bash
/hr/dashboard
/hr/positions
/hr/candidates
/hr/chatbot/:positionId
```

### Candidate

``` bash
/careers
/candidate/cv
/candidate/chatbot
```

------------------------------------------------------------------------

## 4. HR Pages

## 4.1 Position Page

Columns: - id - job title - active (toggle switch) - minimum fit score
(+ edit pen icon) - candidate apply - HR upload - open date - close
date - action

Actions: - View JD - View candidates apply - View HR uploads - Chat
icon - Edit icon

Edit actions: - Update Position - name - language - level - upload JD
file - Delete Position

------------------------------------------------------------------------

## 4.2 Candidate Page

Columns: - id - name - job title - application date - type - stage -
interview schedule - score - reason for match - action

Filters: - search name - stage - type - position - date range

Actions by stage: - APPLIED - Schedule interview - Reject -
INTERVIEW_SCHEDULED - Reschedule - INTERVIEWED - Offer - Reject -
OFFER - Accept by candidate - Reject by candidate

Extra actions: - View CV - View analysis details

------------------------------------------------------------------------

## 4.3 HR Chatbot Page

Features: - Chat history sidebar - New chat - Mode select: - Internal -
External - Chat streaming UI - Markdown render

------------------------------------------------------------------------

## 5. Candidate Portal

## 5.1 Company Career Page

Sections: - Company intro - Active positions - Upload CV - Candidate
chatbot

CV Card: - name - email - upload date - stage - status - applied
positions

Actions: - Update CV - Delete CV - View CV

Position card: - title - posted date - view JD - apply - chatbot icon

------------------------------------------------------------------------

## 6. Shared Components

Required: - AppTable - Upload popup - Delete warning popup - Loading
skeleton - Empty state - File upload box - File preview - Stage tag -
Score badge - Notification toast - Protected route - SSE streaming
component

------------------------------------------------------------------------

## 7. Build Order

### Phase 1

-   setup project
-   routing
-   layout
-   auth
-   axios
-   query client

### Phase 2

-   shared components

### Phase 3

-   HR pages
    -   Position
    -   Candidate
    -   Chatbot

### Phase 4

-   Candidate portal

### Phase 5

-   dashboard
-   responsive
-   polish

------------------------------------------------------------------------

## 8. Dashboard Page

Widgets: - total positions - active positions - total candidates -
accepted - rejected

Charts: - candidates by stage - top positions by applicants

------------------------------------------------------------------------

## 9. Timeline

### Week 1

-   foundation

### Week 2

-   position + candidate pages

### Week 3

-   chatbot + candidate portal

### Week 4

-   dashboard + polish
